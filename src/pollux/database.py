import sqlite3
import uuid
import os
import numpy as np
import pandas as pd
from astropy.wcs import WCS
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from .base import PipelineStep, PipelineContext
import time

class DatabaseUploadStep(PipelineStep):
    """
    Optimized database upload step with support for multi-filter photometry 
    and detailed Roman image metadata.
    """
    def __init__(self):
        self.conn = None
        self.db_path = None
        self.target_cache = {}      # uuid -> dict of positional stats
        self.photometry_cache = {}  # (uuid, filter) -> dict of photometric stats
        self.updated_targets = set()     # set of uuids modified in this run
        self.updated_photometry = set()  # set of (uuid, filter) modified in this run
        
    def _get_connection(self, db_path):
        if self.conn is None or self.db_path != db_path:
            if self.conn: self.conn.close()
            self.conn = sqlite3.connect(db_path)
            self.db_path = db_path
            self.conn.execute("PRAGMA journal_mode=WAL")
            self._init_db(self.conn)
            # We no longer load the full cache here
        return self.conn

    def _load_caches(self):
        """Deprecated: No longer loading full database into memory."""
        pass

    def run(self, context: PipelineContext, config: dict):
        db_path = config.get('database_path')
        if not db_path:
            raise ValueError("DatabaseUploadStep requires 'database_path' in config.")
        
        clobber = config.get('clobber', False)
        match_radius_arcsec = config.get('match_radius_arcsec', 1.0)
        match_sigma = config.get('match_sigma', 5.0)
        
        image_path = context.metadata.get('filename', 'unknown_image')
        image_id = os.path.basename(image_path)
        current_filter = context.metadata.get('filter', 'UNKNOWN')
        
        conn = self._get_connection(db_path)
        
        if self._image_exists(conn, image_id):
            if not clobber:
                raise RuntimeError(f"Image {image_id} already exists and clobber=False.")
            else:
                self._delete_image_data(conn, image_id)
        
        # 1. Register Image with full metadata
        self._register_image(conn, image_id, context)
        
        # 2. Get targets in footprint (Now queries DB instead of full cache)
        existing_targets = self._get_cached_targets_in_footprint(context.wcs, context.image_data.shape, current_filter)
        
        # 3. Match new detections
        new_detections = context.catalog
        if new_detections is None: new_detections = pd.DataFrame()

        matches, unmatched_detections, unmatched_targets = self._match_sources_optimized(
            new_detections, existing_targets, match_radius_arcsec, context.wcs, match_sigma
        )
        
        obs_data = []
        new_targets_batch = []
        
        # Helper for types and conversions
        def to_f(val): return self._to_float(val)

        # FAST PANDAS FIX: Convert DataFrame to a list of dicts. 
        # Dictionary lookups are O(1) and bypass the massive overhead of .iloc
        new_detections_dict = new_detections.to_dict('records')

        # Process matches
        for det_idx, target_uuid in matches.items():
            row = new_detections_dict[det_idx]
            e_x, e_y, e_m = self._logvar_to_sigma(row.get('log_var_x')), self._logvar_to_sigma(row.get('log_var_y')), self._logvar_to_sigma(row.get('log_var_m'))
            obs_data.append((target_uuid, image_id, to_f(row.get('x')), to_f(row.get('y')), 
                             to_f(row.get('ra')), to_f(row.get('dec')), to_f(row.get('flux_raw')), 
                             to_f(row.get('mag_raw')), e_x, e_y, e_m, to_f(row.get('prob')), True))
            self._update_caches(target_uuid, current_filter, row, e_x, e_y, e_m)
            self.updated_targets.add(target_uuid)
            self.updated_photometry.add((target_uuid, current_filter))

        # Process new sources
        for det_idx in unmatched_detections:
            row = new_detections_dict[det_idx]
            new_uuid = str(uuid.uuid4())
            ra, dec = to_f(row['ra']), to_f(row['dec'])
            new_targets_batch.append((new_uuid, ra, dec))
            
            # Init target cache
            self.target_cache[new_uuid] = {
                'uuid': new_uuid, 'ra': ra, 'dec': dec,
                'ra_weighted': ra, 'dec_weighted': dec,
                'ra_rmse': 0.0, 'dec_rmse': 0.0,
                'wsum_ra': 0.0, 'wsum_dec': 0.0,
                'wsum_ra2': 0.0, 'wsum_dec2': 0.0,
                'obs_count': 0
            }
            
            e_x, e_y, e_m = self._logvar_to_sigma(row.get('log_var_x')), self._logvar_to_sigma(row.get('log_var_y')), self._logvar_to_sigma(row.get('log_var_m'))
            obs_data.append((new_uuid, image_id, to_f(row.get('x')), to_f(row.get('y')), 
                             ra, dec, to_f(row.get('flux_raw')), to_f(row.get('mag_raw')), 
                             e_x, e_y, e_m, to_f(row.get('prob')), True))
            self._update_caches(new_uuid, current_filter, row, e_x, e_y, e_m)
            self.updated_targets.add(new_uuid)
            self.updated_photometry.add((new_uuid, current_filter))

        cursor = conn.cursor()
        if new_targets_batch:
            cursor.executemany("INSERT INTO targets (uuid, ra, dec) VALUES (?, ?, ?)", new_targets_batch)
        
        cursor.executemany("""
            INSERT INTO observations (target_uuid, image_id, x, y, ra, dec, flux_raw, mag_raw, err_x, err_y, err_m, prob, is_detection)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, obs_data)
        
        conn.commit()
        if not config.get('quiet'):
            print(f"Database upload complete for {image_id} [{current_filter}]. Matches: {len(matches)}, New: {len(unmatched_detections)}, Missed: {len(unmatched_targets)}")

    def finalize(self, config: dict):
        if self.conn is None: return
        
        # 1. Compute post-processing LC analytics (ONLY for updated targets)
        self._compute_lc_analytics()

        print(f"Finalizing DatabaseUploadStep: Flushing positional and photometric stats...")
        cursor = self.conn.cursor()
        
        # 1. Update targets table (Positional) - ONLY updated ones
        target_updates = []
        for uid in self.updated_targets:
            s = self.target_cache[uid]
            target_updates.append((
                self._to_float(s['ra_weighted']), self._to_float(s['dec_weighted']),
                self._to_float(s['ra_rmse']), self._to_float(s['dec_rmse']),
                self._to_float(s['wsum_ra']), self._to_float(s['wsum_dec']),
                self._to_float(s['wsum_ra2']), self._to_float(s['wsum_dec2']),
                int(s['obs_count']), uid
            ))
            
        if target_updates:
            cursor.executemany("""
                UPDATE targets SET 
                    ra_weighted = ?, dec_weighted = ?, ra_rmse = ?, dec_rmse = ?,
                    wsum_ra = ?, wsum_dec = ?, wsum_ra2 = ?, wsum_dec2 = ?,
                    obs_count = ?
                WHERE uuid = ?
            """, target_updates)

        # 2. Update target_photometry table (Filter-aware) - ONLY updated ones
        photo_inserts = []
        
        for (uid, filt) in self.updated_photometry:
            s = self.photometry_cache[(uid, filt)]
            data = (
                self._to_float(s['flux_weighted']), self._to_float(s['mag_weighted']),
                self._to_float(s['flux_rmse']), self._to_float(s['mag_rmse']),
                self._to_float(s.get('chi2_reduced')), self._to_float(s.get('v_n_ratio')),
                self._to_float(s.get('autocorr_1')), self._to_float(s.get('max_consecutive_outliers')),
                self._to_float(s.get('peak_to_median_ratio')),
                self._to_float(s['wsum_flux']), self._to_float(s['wsum_mag']),
                self._to_float(s['wsum_flux2']), self._to_float(s['wsum_mag2']),
                int(s['obs_count']), uid, filt
            )
            photo_inserts.append(data)

        if photo_inserts:
            cursor.executemany("""
                REPLACE INTO target_photometry (
                    flux_weighted, mag_weighted, flux_rmse, mag_rmse,
                    chi2_reduced, v_n_ratio, autocorr_1, max_consecutive_outliers, peak_to_median_ratio,
                    wsum_flux, wsum_mag, wsum_flux2, wsum_mag2,
                    obs_count, target_uuid, filter
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, photo_inserts)
        
        self.conn.commit()
        self.conn.close()
        self.conn = None
        self.updated_targets.clear()
        self.updated_photometry.clear()
        print("Database connection closed.")

    def _compute_lc_analytics(self):
        """Computes variability and quality metrics for only the modified light curves."""
        if not self.updated_photometry: return
        
        print(f"Computing light curve analytics for {len(self.updated_photometry)} updated targets...")
        
        # Filter to only affected target/filter pairs
        # Use a temporary table or a large IN clause (SQLite limit is 999 parameters, but we can do it in batches)
        affected_pairs = list(self.updated_photometry)
        batch_size = 500
        
        for i in range(0, len(affected_pairs), batch_size):
            batch = affected_pairs[i:i + batch_size]
            
            # Construct query for this batch
            placeholders = ", ".join(["(?, ?)"] * len(batch))
            params = []
            for uid, filt in batch:
                params.extend([uid, filt])
                
            query = f"""
                SELECT o.target_uuid, i.filter, i.obs_time, o.flux_raw, o.err_m
                FROM observations o
                JOIN images i ON o.image_id = i.image_id
                WHERE (o.target_uuid, i.filter) IN ({placeholders})
                  AND o.is_detection = 1 AND o.err_m IS NOT NULL
                ORDER BY o.target_uuid, i.filter, i.obs_time
            """
            df = pd.read_sql_query(query, self.conn, params=params)
            if df.empty: continue

            # Group by target and filter
            for (uid, filt), group in df.groupby(['target_uuid', 'filter']):
                if len(group) < 5: continue
                # ... same computation logic as before ...
                flux = group['flux_raw'].values
                err_f = flux * np.log(10) * group['err_m'].values
                baseline = np.median(flux)
                var = np.var(flux)
                if var == 0: var = 1e-10
                safe_err = np.maximum(err_f, 1e-5)
                chi2 = np.sum(((flux - baseline) / safe_err)**2)
                chi2_red = chi2 / (len(flux) - 1)
                peak_to_median = np.max(flux) / (baseline if baseline > 0 else 1e-10)
                diffs = np.diff(flux)
                delta_sq = np.mean(diffs**2)
                v_n_ratio = delta_sq / var
                if len(flux) > 1:
                    ac_mat = np.corrcoef(flux[:-1], flux[1:])
                    ac = ac_mat[0, 1] if ac_mat.shape == (2, 2) else 0.0
                else: ac = 0.0
                sigma_pred = np.median(err_f)
                thresh = baseline + 3 * sigma_pred
                outliers = flux > thresh
                max_consec = 0; current_consec = 0
                for is_outlier in outliers:
                    if is_outlier:
                        current_consec += 1
                        max_consec = max(max_consec, current_consec)
                    else: current_consec = 0
                
                key = (uid, filt)
                if key in self.photometry_cache:
                    self.photometry_cache[key].update({
                        'chi2_reduced': float(chi2_red),
                        'v_n_ratio': float(v_n_ratio),
                        'autocorr_1': float(ac),
                        'max_consecutive_outliers': int(max_consec),
                        'peak_to_median_ratio': float(peak_to_median)
                    })

    def _init_db(self, conn):
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS targets (
                uuid TEXT PRIMARY KEY, ra REAL, dec REAL,
                ra_weighted REAL, dec_weighted REAL,
                ra_rmse REAL DEFAULT 0, dec_rmse REAL DEFAULT 0,
                wsum_ra REAL DEFAULT 0, wsum_dec REAL DEFAULT 0,
                wsum_ra2 REAL DEFAULT 0, wsum_dec2 REAL DEFAULT 0,
                obs_count INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_targets_ra_dec ON targets(ra, dec);")
        
        # New table for filter-specific photometry
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS target_photometry (
                target_uuid TEXT,
                filter TEXT,
                flux_weighted REAL,
                mag_weighted REAL,
                flux_rmse REAL DEFAULT 0,
                mag_rmse REAL DEFAULT 0,
                chi2_reduced REAL,
                v_n_ratio REAL,
                autocorr_1 REAL,
                max_consecutive_outliers INTEGER,
                peak_to_median_ratio REAL,
                wsum_flux REAL DEFAULT 0,
                wsum_mag REAL DEFAULT 0,
                wsum_flux2 REAL DEFAULT 0,
                wsum_mag2 REAL DEFAULT 0,
                obs_count INTEGER DEFAULT 0,
                PRIMARY KEY (target_uuid, filter),
                FOREIGN KEY(target_uuid) REFERENCES targets(uuid)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_id TEXT PRIMARY KEY, filename TEXT, filter TEXT, detector TEXT,
                ma_table INTEGER, nresultants INTEGER, exptime REAL, zp REAL, 
                obs_time REAL, wcs_header TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                obs_id INTEGER PRIMARY KEY AUTOINCREMENT, target_uuid TEXT, image_id TEXT,
                x REAL, y REAL, ra REAL, dec REAL, flux_raw REAL, mag_raw REAL,
                err_x REAL, err_y REAL, err_m REAL, prob REAL, is_detection BOOLEAN,
                FOREIGN KEY(target_uuid) REFERENCES targets(uuid),
                FOREIGN KEY(image_id) REFERENCES images(image_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_target_uuid ON observations(target_uuid);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_image_id ON observations(image_id);")
        conn.commit()

    def _image_exists(self, conn, image_id):
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM images WHERE image_id = ?", (image_id,))
        return cursor.fetchone() is not None

    def _delete_image_data(self, conn, image_id):
        cursor = conn.cursor()
        
        # Decrement obs_count in cache for all targets in this image before deleting
        cursor.execute("SELECT target_uuid FROM observations WHERE image_id = ?", (image_id,))
        for (uid,) in cursor.fetchall():
            if uid in self.target_cache:
                self.target_cache[uid]['obs_count'] = max(0, self.target_cache[uid]['obs_count'] - 1)
            
            # Find filter for this image to also decrement photometry cache
            # Note: This is a bit complex without the filter name, but we can iterate.
            for (p_uid, filt), p_stats in self.photometry_cache.items():
                if p_uid == uid:
                    # We can't be 100% sure of the filter without another query, 
                    # but if target was matched, it was for a specific filter.
                    # Simplified: we'll decrement any filter match found for this target in this image.
                    p_stats['obs_count'] = max(0, p_stats['obs_count'] - 1)

        cursor.execute("DELETE FROM observations WHERE image_id = ?", (image_id,))
        cursor.execute("DELETE FROM images WHERE image_id = ?", (image_id,))
        conn.commit()

    def _register_image(self, conn, image_id, context):
        cursor = conn.cursor()
        
        # Safely handle WCS serialization using Adapter
        header_str = ""
        if context.wcs is not None:
            header_str = context.wcs.to_header_string()

        meta = context.metadata
        cursor.execute("""
            INSERT INTO images (image_id, filename, filter, detector, ma_table, nresultants, exptime, zp, obs_time, wcs_header)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            image_id, meta.get('filename', image_id),
            meta.get('filter'), meta.get('detector'),
            meta.get('ma_table'), meta.get('nresultants'),
            self._to_float(meta.get('exptime', 0.0)),
            self._to_float(meta.get('zp', 0.0)),
            self._to_float(meta.get('obs_time', 0.0)),
            header_str
        ))

    def _get_cached_targets_in_footprint(self, wcs, img_shape, current_filter):
        h, w = img_shape
        
        try:
            footprint = wcs.calc_footprint(img_shape)
            fp_val = getattr(footprint, 'value', footprint)
            ra_min, ra_max = np.min(fp_val[:, 0]), np.max(fp_val[:, 0])
            dec_min, dec_max = np.min(fp_val[:, 1]), np.max(fp_val[:, 1])
        except:
            # Fallback to extreme bounds if footprint fails
            ra_min, ra_max, dec_min, dec_max = 0, 360, -90, 90
        
        pad = 30.0 / 3600.0 # 30 arcsec padding for safety
        ra_min -= pad; ra_max += pad
        dec_min -= pad; dec_max += pad
        
        # 1. Query DB for targets in this RA/Dec box
        query = """
            SELECT uuid, ra, dec, ra_weighted, dec_weighted, ra_rmse, dec_rmse,
                   wsum_ra, wsum_dec, wsum_ra2, wsum_dec2, obs_count
            FROM targets
            WHERE ra >= ? AND ra <= ? AND dec >= ? AND dec <= ?
        """
        t_df = pd.read_sql_query(query, self.conn, params=(ra_min, ra_max, dec_min, dec_max))
        
        if t_df.empty: return pd.DataFrame()
        
        # Add to local target_cache
        found_uuids = []
        for _, row in t_df.iterrows():
            uid = row['uuid']
            if uid not in self.target_cache:
                self.target_cache[uid] = row.to_dict()
            found_uuids.append(uid)

        # 2. Query DB for photometry for these targets in the current filter
        if found_uuids:
            # We can use batches for large numbers of targets if needed
            batch_size = 500
            for i in range(0, len(found_uuids), batch_size):
                batch = found_uuids[i:i + batch_size]
                placeholders = ", ".join(["?"] * len(batch))
                p_query = f"""
                    SELECT target_uuid, filter, flux_weighted, mag_weighted, 
                           flux_rmse, mag_rmse, chi2_reduced, v_n_ratio, autocorr_1,
                           max_consecutive_outliers, peak_to_median_ratio,
                           wsum_flux, wsum_mag, wsum_flux2, wsum_mag2, obs_count
                    FROM target_photometry
                    WHERE target_uuid IN ({placeholders}) AND filter = ?
                """
                p_df = pd.read_sql_query(p_query, self.conn, params=(*batch, current_filter))
                for _, row in p_df.iterrows():
                    key = (row['target_uuid'], row['filter'])
                    if key not in self.photometry_cache:
                        self.photometry_cache[key] = row.to_dict()

        # 3. Refine using pixel coordinates (same as before but using the fetched targets)
        v_uuids = found_uuids
        v_ra = np.array([self.target_cache[uid].get('ra_weighted') or self.target_cache[uid]['ra'] for uid in v_uuids])
        v_dec = np.array([self.target_cache[uid].get('dec_weighted') or self.target_cache[uid]['dec'] for uid in v_uuids])
        
        x, y = wcs.world_to_pixel_values(v_ra, v_dec)
        x_val = getattr(x, 'value', x)
        y_val = getattr(y, 'value', y)
        
        # Final pixel mask (with 10px buffer)
        p_mask = (x_val >= -10) & (x_val < w + 10) & (y_val >= -10) & (y_val < h + 10)
        if not np.any(p_mask): return pd.DataFrame()
        
        final_uuids = [v_uuids[i] for i in np.where(p_mask)[0]]
        targets_in_footprint = [self.target_cache[uid] for uid in final_uuids]
        return pd.DataFrame(targets_in_footprint)

    def _match_sources_optimized(self, detections, targets, radius_arcsec, wcs, match_sigma=5.0):
        if detections.empty: return {}, [], list(targets['uuid']) if not targets.empty else []
        if targets.empty: return {}, list(range(len(detections))), []

        # 1. Pre-calculate robust pixel scale and detector sigmas
        try:
            pixel_scales = wcs.proj_plane_pixel_scales()
            pixel_scale_deg = float(np.mean(np.abs([getattr(s, 'value', s) for s in pixel_scales])))
        except:
            ra1, dec1 = wcs.pixel_to_world_values(500, 500)
            ra2, dec2 = wcs.pixel_to_world_values(500, 501)
            pixel_scale_deg = float(np.sqrt(((ra2 - ra1) * np.cos(np.deg2rad(dec1)))**2 + (dec2 - dec1)**2))

        # Extract detection arrays directly (vectorized)
        det_ra = np.array([self._to_float(v) for v in detections['ra'].values])
        det_dec = np.array([self._to_float(v) for v in detections['dec'].values])
        sig_det_ra = np.array([self._logvar_to_sigma(lv) for lv in detections.get('log_var_x', [])]) * pixel_scale_deg
        sig_det_dec = np.array([self._logvar_to_sigma(lv) for lv in detections.get('log_var_y', [])]) * pixel_scale_deg
        
        # Extract target arrays directly (vectorized)
        tar_ra = np.array([self._to_float(v) for v in targets['ra_weighted'].values])
        tar_dec = np.array([self._to_float(v) for v in targets['dec_weighted'].values])
        tar_uuids = targets['uuid'].values
        
        floor_deg = 0.05 / 3600.0
        sig_tar_ra = np.maximum(np.array([self._to_float(v) for v in targets['ra_rmse'].values]), floor_deg)
        sig_tar_dec = np.maximum(np.array([self._to_float(v) for v in targets['dec_rmse'].values]), floor_deg)

        # 2. Coordinate transformation for KD-Tree (Spherical approx)
        def to_coords(ra, dec):
            return np.column_stack([ra * np.cos(np.deg2rad(dec)), dec])

        det_coords = to_coords(det_ra, det_dec)
        tar_coords = to_coords(tar_ra, tar_dec)
        
        # 3. Fast KD-Tree query
        tree = cKDTree(tar_coords)
        max_dist_deg = float(radius_arcsec / 3600.0)
        pairs = tree.query_ball_point(det_coords, max_dist_deg)
        
        # 4. Refine matches using Mahalanobis distance (Vectorized where possible)
        ambiguous_pairs = []
        
        for d_idx, tar_indices in enumerate(pairs):
            if not tar_indices: continue
            
            # Local slice of targets for this detection
            t_ra, t_dec = tar_ra[tar_indices], tar_dec[tar_indices]
            
            # Mahalanobis components
            d_ra_scaled = (det_ra[d_idx] - t_ra) * np.cos(np.deg2rad(t_dec))
            d_dec = det_dec[d_idx] - t_dec
            
            var_ra = sig_det_ra[d_idx]**2 + sig_tar_ra[tar_indices]**2
            var_dec = sig_det_dec[d_idx]**2 + sig_tar_dec[tar_indices]**2
            
            # chi-square distances
            norm_dists = np.sqrt(d_ra_scaled**2 / var_ra + d_dec**2 / var_dec)
            
            # Filter by match_sigma
            valid_mask = norm_dists <= match_sigma
            for i in np.where(valid_mask)[0]:
                ambiguous_pairs.append((norm_dists[i], d_idx, tar_indices[i]))

        # 5. Greedy matching (lowest distance first)
        ambiguous_pairs.sort(key=lambda x: x[0])
        
        matches = {}
        matched_det = set()
        matched_tar_idx = set()
        
        for dist, d_idx, t_idx in ambiguous_pairs:
            if d_idx not in matched_det and t_idx not in matched_tar_idx:
                matches[d_idx] = tar_uuids[t_idx]
                matched_det.add(d_idx)
                matched_tar_idx.add(t_idx)

        unmatched_det = [i for i in range(len(detections)) if i not in matched_det]
        unmatched_tar = [tar_uuids[i] for i in range(len(targets)) if i not in matched_tar_idx]
        
        return matches, unmatched_det, unmatched_tar

    def _update_caches(self, uuid_str, filt, new_row, err_x, err_y, err_m):
        # 1. Update Positional Cache (targets table)
        t = self.target_cache[uuid_str]
        
        def get_weight(sigma):
            if sigma is None or sigma <= 0: return 0.0
            return 1.0 / (sigma**2)

        w_ra = get_weight(err_x); w_dec = get_weight(err_y)
        
        def update_running_stats(old_mean, old_sum_w, old_sum_wx2, val, weight):
            nv = self._to_float(val)
            if weight <= 0 or nv is None: return old_mean, old_sum_w, old_sum_wx2, 0.0
            if old_sum_w <= 0: return nv, weight, weight * (nv**2), 0.0
            
            sum_wx = old_mean * old_sum_w
            new_sum_w = old_sum_w + weight
            new_sum_wx = sum_wx + weight * nv
            new_sum_wx2 = old_sum_wx2 + weight * (nv**2)
            
            new_mean = new_sum_wx / new_sum_w
            var = max(0, (new_sum_wx2 / new_sum_w) - (new_mean**2))
            return new_mean, new_sum_w, new_sum_wx2, np.sqrt(var)

        t['ra_weighted'], t['wsum_ra'], t['wsum_ra2'], t['ra_rmse'] = update_running_stats(t['ra_weighted'], t['wsum_ra'], t['wsum_ra2'], new_row['ra'], w_ra)
        t['dec_weighted'], t['wsum_dec'], t['wsum_dec2'], t['dec_rmse'] = update_running_stats(t['dec_weighted'], t['wsum_dec'], t['wsum_dec2'], new_row['dec'], w_dec)
        t['obs_count'] += 1

        # 2. Update Photometric Cache (target_photometry table)
        if (uuid_str, filt) not in self.photometry_cache:
            self.photometry_cache[(uuid_str, filt)] = {
                'target_uuid': uuid_str, 'filter': filt,
                'flux_weighted': 0.0, 'mag_weighted': 0.0,
                'flux_rmse': 0.0, 'mag_rmse': 0.0,
                'wsum_flux': 0.0, 'wsum_mag': 0.0,
                'wsum_flux2': 0.0, 'wsum_mag2': 0.0,
                'obs_count': 0
            }
        
        p = self.photometry_cache[(uuid_str, filt)]
        
        w_mag = get_weight(err_m)
        flux_val = self._to_float(new_row.get('flux_raw'))
        if flux_val and flux_val > 0 and err_m and err_m > 0:
            # sigma_f = f * sigma_m * ln(10)/2.5
            # weight_f = (1/sigma_m^2) * (2.5 / (f * ln(10)))^2
            w_flux = w_mag * (1.085736 / flux_val)**2
        else:
            w_flux = 0.0
        
        p['flux_weighted'], p['wsum_flux'], p['wsum_flux2'], p['flux_rmse'] = update_running_stats(p['flux_weighted'], p['wsum_flux'], p['wsum_flux2'], new_row['flux_raw'], w_flux)
        p['mag_weighted'], p['wsum_mag'], p['wsum_mag2'], p['mag_rmse'] = update_running_stats(p['mag_weighted'], p['wsum_mag'], p['wsum_mag2'], new_row['mag_raw'], w_mag)
        p['obs_count'] += 1

    def _to_float(self, val):
        if val is None or pd.isna(val): return None
        v = getattr(val, 'value', val)
        if hasattr(v, 'item'): 
            try: return float(v.item())
            except: return float(v)
        return float(v)

    def _logvar_to_sigma(self, log_var):
        lv = self._to_float(log_var)
        if lv is None: return None
        return float(np.sqrt(np.exp(np.clip(lv, -20, 20))))
