import sqlite3
import uuid
import os
import numpy as np
import pandas as pd
from astropy.wcs import WCS
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from .base import PipelineStep, PipelineContext

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
        
    def _get_connection(self, db_path):
        if self.conn is None or self.db_path != db_path:
            if self.conn: self.conn.close()
            self.conn = sqlite3.connect(db_path)
            self.db_path = db_path
            self.conn.execute("PRAGMA journal_mode=WAL")
            self._init_db(self.conn)
            self._load_caches()
        return self.conn

    def _load_caches(self):
        """Loads existing targets and filter-specific photometry into memory."""
        # 1. Load Positional Stats
        t_df = pd.read_sql_query("""
            SELECT uuid, ra, dec, ra_weighted, dec_weighted, ra_rmse, dec_rmse,
                   wsum_ra, wsum_dec, wsum_ra2, wsum_dec2, obs_count
            FROM targets
        """, self.conn)
        self.target_cache = {row['uuid']: row.to_dict() for _, row in t_df.iterrows()}
        
        # 2. Load Photometric Stats
        p_df = pd.read_sql_query("""
            SELECT target_uuid, filter, flux_weighted, mag_weighted, 
                   flux_rmse, mag_rmse, wsum_flux, wsum_mag, 
                   wsum_flux2, wsum_mag2, obs_count
            FROM target_photometry
        """, self.conn)
        self.photometry_cache = {
            (row['target_uuid'], row['filter']): row.to_dict() 
            for _, row in p_df.iterrows()
        }
        
        print(f"Loaded {len(self.target_cache)} targets and {len(self.photometry_cache)} photometry records.")

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
        
        # 2. Get targets in footprint
        existing_targets = self._get_cached_targets_in_footprint(context.wcs, context.image_data.shape)
        
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

        # FAST ASTROPY FIX: Vectorize the WCS Transformation
        if unmatched_targets:
            missed_uuids = list(unmatched_targets)
            missed_ras = [self.target_cache[uid]['ra_weighted'] for uid in missed_uuids]
            missed_decs = [self.target_cache[uid]['dec_weighted'] for uid in missed_uuids]
            
            # Transform all 13,000+ coordinates in one swift C-optimized sweep
            missed_x, missed_y = context.wcs.world_to_pixel_values(missed_ras, missed_decs)
            
            # Ensure arrays are 1D (Astropy returns 0D if there's only 1 target)
            missed_x = np.atleast_1d(missed_x)
            missed_y = np.atleast_1d(missed_y)
            
            for i, target_uuid in enumerate(missed_uuids):
                obs_data.append((target_uuid, image_id, to_f(missed_x[i]), to_f(missed_y[i]), 
                                 to_f(missed_ras[i]), to_f(missed_decs[i]), None, None, None, None, None, 0.0, False))
                self.target_cache[target_uuid]['obs_count'] += 1

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
        
        # 1. Compute post-processing LC analytics
        self._compute_lc_analytics()

        print(f"Finalizing DatabaseUploadStep: Flushing positional and photometric stats...")
        cursor = self.conn.cursor()
        
        # 1. Update targets table (Positional)
        target_updates = []
        for uid, s in self.target_cache.items():
            target_updates.append((
                self._to_float(s['ra_weighted']), self._to_float(s['dec_weighted']),
                self._to_float(s['ra_rmse']), self._to_float(s['dec_rmse']),
                self._to_float(s['wsum_ra']), self._to_float(s['wsum_dec']),
                self._to_float(s['wsum_ra2']), self._to_float(s['wsum_dec2']),
                int(s['obs_count']), uid
            ))
            
        cursor.executemany("""
            UPDATE targets SET 
                ra_weighted = ?, dec_weighted = ?, ra_rmse = ?, dec_rmse = ?,
                wsum_ra = ?, wsum_dec = ?, wsum_ra2 = ?, wsum_dec2 = ?,
                obs_count = ?
            WHERE uuid = ?
        """, target_updates)

        # 2. Update target_photometry table (Filter-aware)
        photo_inserts = []
        
        for (uid, filt), s in self.photometry_cache.items():
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
        print("Database connection closed.")

    def _compute_lc_analytics(self):
        """Computes variability and quality metrics for all light curves in the DB."""
        print("Computing light curve analytics for all targets...")
        
        # Pull all relevant data in one big join
        query = """
            SELECT o.target_uuid, i.filter, i.obs_time, o.flux_raw, o.err_m
            FROM observations o
            JOIN images i ON o.image_id = i.image_id
            WHERE o.is_detection = 1
            ORDER BY o.target_uuid, i.filter, i.obs_time
        """
        df = pd.read_sql_query(query, self.conn)
        if df.empty: return

        # Group by target and filter
        for (uid, filt), group in df.groupby(['target_uuid', 'filter']):
            if len(group) < 5: continue
            
            flux = group['flux_raw'].values
            # Convert mag error to flux error proxy: sigma_f = f * ln(10) * sigma_m
            err_f = flux * np.log(10) * group['err_m'].values
            
            # 1. Robust Baseline (Median is less biased by the spike than the mean)
            baseline = np.median(flux)
            var = np.var(flux)
            if var == 0: var = 1e-10
            
            # 2. Reduced Chi-squared (constant flux fit relative to median)
            safe_err = np.maximum(err_f, 1e-5)
            chi2 = np.sum(((flux - baseline) / safe_err)**2)
            chi2_red = chi2 / (len(flux) - 1)
            
            # 3. Peak-to-Median Ratio (dynamic range of the event)
            peak_to_median = np.max(flux) / (baseline if baseline > 0 else 1e-10)
            
            # 4. Von Neumann Ratio (eta) - measures smoothness
            diffs = np.diff(flux)
            delta_sq = np.mean(diffs**2)
            v_n_ratio = delta_sq / var
            
            # 5. Lag-1 Autocorrelation
            if len(flux) > 1:
                ac_mat = np.corrcoef(flux[:-1], flux[1:])
                ac = ac_mat[0, 1] if ac_mat.shape == (2, 2) else 0.0
                if np.isnan(ac): ac = 0.0
            else:
                ac = 0.0
                
            # 6. Alert: Max consecutive outliers (> 3 sigma above baseline)
            # We use the median predicted error as the sigma threshold
            sigma_pred = np.median(err_f)
            thresh = baseline + 3 * sigma_pred
            
            outliers = flux > thresh
            max_consec = 0
            current_consec = 0
            for is_outlier in outliers:
                if is_outlier:
                    current_consec += 1
                    max_consec = max(max_consec, current_consec)
                else:
                    current_consec = 0
            
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
        
        # Safely handle WCS serialization
        header_str = ""
        if context.wcs is not None:
            if hasattr(context.wcs, 'to_header'):
                # Standard FITS WCS
                header_str = context.wcs.to_header().tostring()
            else:
                # It's a GWCS (ASDF)
                header_str = "GWCS_OBJECT"

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

    def _get_cached_targets_in_footprint(self, wcs, img_shape):
        if not self.target_cache: return pd.DataFrame()
        h, w = img_shape
        try:
            footprint = wcs.calc_footprint()
            fp_val = getattr(footprint, 'value', footprint)
            ra_min, ra_max = np.min(fp_val[:, 0]), np.max(fp_val[:, 0])
            dec_min, dec_max = np.min(fp_val[:, 1]), np.max(fp_val[:, 1])
        except:
            ra_min, ra_max, dec_min, dec_max = 0, 360, -90, 90
        
        pad = 10.0 / 3600.0
        ra_min -= pad; ra_max += pad
        dec_min -= pad; dec_max += pad
        
        targets_in_footprint = []
        for t in self.target_cache.values():
            if ra_min <= t['ra_weighted'] <= ra_max and dec_min <= t['dec_weighted'] <= dec_max:
                targets_in_footprint.append(t)
        
        if not targets_in_footprint: return pd.DataFrame()
        df = pd.DataFrame(targets_in_footprint)
        x, y = wcs.world_to_pixel_values(df['ra_weighted'].values, df['dec_weighted'].values)
        x_val = getattr(x, 'value', x)
        y_val = getattr(y, 'value', y)
        mask = (x_val >= -5) & (x_val < w + 5) & (y_val >= -5) & (y_val < h + 5)
        return df[mask]

    def _match_sources_optimized(self, detections, targets, radius_arcsec, wcs, match_sigma=5.0):
        if detections.empty: return {}, [], list(targets['uuid']) if not targets.empty else []
        if targets.empty: return {}, list(range(len(detections))), []

        # Robust pixel scale calculation for both FITS WCS and GWCS
        try:
            pixel_scales = wcs.proj_plane_pixel_scales()
            scales_val = [getattr(s, 'value', s) for s in pixel_scales]
            pixel_scale_deg = float(np.mean(np.abs(scales_val)))
        except AttributeError:
            # GWCS fallback: Empirically calculate the pixel scale using two adjacent pixels
            ra1, dec1 = wcs.pixel_to_world_values(500, 500)
            ra2, dec2 = wcs.pixel_to_world_values(500, 501)

            # Apply cosine correction for RA to get true angular distance
            d_ra = (ra2 - ra1) * np.cos(np.deg2rad(dec1))
            d_dec = dec2 - dec1
            pixel_scale_deg = float(np.sqrt(d_ra**2 + d_dec**2))

        sig_det_ra = np.array([self._logvar_to_sigma(lv) for lv in detections.get('log_var_x', [])]) * pixel_scale_deg
        sig_det_dec = np.array([self._logvar_to_sigma(lv) for lv in detections.get('log_var_y', [])]) * pixel_scale_deg
        floor_deg = 0.05 / 3600.0
        sig_tar_ra = np.array([self._to_float(v) for v in targets['ra_rmse'].values])
        sig_tar_dec = np.array([self._to_float(v) for v in targets['dec_rmse'].values])
        sig_tar_ra = np.maximum(sig_tar_ra, floor_deg)
        sig_tar_dec = np.maximum(sig_tar_dec, floor_deg)

        def get_series_val(df, col):
            vals = df[col].values
            return getattr(vals, 'value', vals)

        def to_coords_pure(df, ra_col='ra', dec_col='dec'):
            ra = np.array([self._to_float(v) for v in df[ra_col].values])
            dec = np.array([self._to_float(v) for v in df[dec_col].values])
            return np.column_stack([ra * np.cos(np.deg2rad(dec)), dec])

        det_coords_val = to_coords_pure(detections)
        tar_coords_val = to_coords_pure(targets, 'ra_weighted', 'dec_weighted')
        
        tree = cKDTree(tar_coords_val)
        max_dist_deg = float(radius_arcsec / 3600.0)
        pairs = tree.query_ball_point(det_coords_val, max_dist_deg)
        
        matches = {}
        matched_det = set()
        matched_tar_idx = set()
        
        for d_idx, tar_indices in enumerate(pairs):
            if len(tar_indices) == 1:
                t_idx = tar_indices[0]
                if t_idx not in matched_tar_idx:
                    det_ra = self._to_float(detections.iloc[d_idx]['ra'])
                    det_dec = self._to_float(detections.iloc[d_idx]['dec'])
                    tar_ra = self._to_float(targets.iloc[t_idx]['ra_weighted'])
                    tar_dec = self._to_float(targets.iloc[t_idx]['dec_weighted'])
                    
                    d_ra_val = det_ra - tar_ra
                    d_dec_val = det_dec - tar_dec
                    d_ra_scaled = float(d_ra_val * np.cos(np.deg2rad(tar_dec)))
                    d_dec_match = float(d_dec_val)
                    
                    var_ra = float(sig_det_ra[d_idx]**2 + sig_tar_ra[t_idx]**2)
                    var_dec = float(sig_det_dec[d_idx]**2 + sig_tar_dec[t_idx]**2)
                    norm_dist = np.sqrt(d_ra_scaled**2 / var_ra + d_dec_match**2 / var_dec)
                    
                    if norm_dist <= match_sigma:
                        matches[d_idx] = targets.iloc[t_idx]['uuid']
                        matched_det.add(d_idx); matched_tar_idx.add(t_idx)

        remaining_det = [i for i in range(len(detections)) if i not in matched_det and len(pairs[i]) > 0]
        
        if remaining_det:
            ambiguous_pairs = []
            
            # Only evaluate pairs already identified by the KD-Tree (localized)
            for d_idx in remaining_det:
                for t_idx in pairs[d_idx]:
                    if t_idx not in matched_tar_idx:
                        det_ra = self._to_float(detections.iloc[d_idx]['ra'])
                        det_dec = self._to_float(detections.iloc[d_idx]['dec'])
                        tar_ra = self._to_float(targets.iloc[t_idx]['ra_weighted'])
                        tar_dec = self._to_float(targets.iloc[t_idx]['dec_weighted'])
                        
                        d_ra_val = det_ra - tar_ra
                        d_dec_val = det_dec - tar_dec
                        d_ra_scaled = float(d_ra_val * np.cos(np.deg2rad(tar_dec)))
                        d_dec_match = float(d_dec_val)
                        
                        var_ra = float(sig_det_ra[d_idx]**2 + sig_tar_ra[t_idx]**2)
                        var_dec = float(sig_det_dec[d_idx]**2 + sig_tar_dec[t_idx]**2)
                        norm_dist = np.sqrt(d_ra_scaled**2 / var_ra + d_dec_match**2 / var_dec)
                        
                        if norm_dist <= match_sigma:
                            ambiguous_pairs.append((norm_dist, d_idx, t_idx))
            
            # Sort by lowest normalized distance first (Greedy match)
            ambiguous_pairs.sort(key=lambda x: x[0])
            
            # Assign best matches first, ignoring ones already snatched up
            for norm_dist, d_idx, t_idx in ambiguous_pairs:
                if d_idx not in matched_det and t_idx not in matched_tar_idx:
                    matches[d_idx] = targets.iloc[t_idx]['uuid']
                    matched_det.add(d_idx)
                    matched_tar_idx.add(t_idx)

        unmatched_det = [i for i in range(len(detections)) if i not in matched_det]
        unmatched_tar = [targets.iloc[i]['uuid'] for i in range(len(targets)) if i not in matched_tar_idx]
        return matches, unmatched_det, unmatched_tar

    def _update_caches(self, uuid_str, filt, new_row, err_x, err_y, err_m):
        # 1. Update Positional Cache (targets table)
        t = self.target_cache[uuid_str]
        
        def get_weight(sigma):
            if sigma is None or sigma <= 0: return 0.0
            return 1.0 / (sigma**2)

        w_ra = get_weight(err_x); w_dec = get_weight(err_y)
        
        def update_stats(old_mean, old_wsum, old_wsum2, val, weight):
            nv = self._to_float(val)
            if weight == 0 or nv is None: return old_mean, old_wsum, old_wsum2, 0.0
            if old_wsum == 0: return nv, weight, weight * (nv**2), 0.0
            new_wsum = old_wsum + weight
            new_wsum2 = old_wsum2 + weight * (nv**2)
            new_mean = (old_mean * old_wsum + nv * weight) / new_wsum
            var = max(0, (new_wsum2 / new_wsum) - (new_mean**2))
            return new_mean, new_wsum, new_wsum2, np.sqrt(var)

        t['ra_weighted'], t['wsum_ra'], t['wsum_ra2'], t['ra_rmse'] = update_stats(t['ra_weighted'], t['wsum_ra'], t['wsum_ra2'], new_row['ra'], w_ra)
        t['dec_weighted'], t['wsum_dec'], t['wsum_dec2'], t['dec_rmse'] = update_stats(t['dec_weighted'], t['wsum_dec'], t['wsum_dec2'], new_row['dec'], w_dec)
        t['obs_count'] += 1

        # 2. Update Photometric Cache (target_photometry table)
        if (uuid_str, filt) not in self.photometry_cache:
            self.photometry_cache[(uuid_str, filt)] = {
                'target_uuid': uuid_str, 'filter': filt,
                'flux_weighted': self._to_float(new_row.get('flux_raw')),
                'mag_weighted': self._to_float(new_row.get('mag_raw')),
                'flux_rmse': 0.0, 'mag_rmse': 0.0,
                'wsum_flux': 0.0, 'wsum_mag': 0.0,
                'wsum_flux2': 0.0, 'wsum_mag2': 0.0,
                'obs_count': 0
            }
        
        p = self.photometry_cache[(uuid_str, filt)]
        w_mag = get_weight(err_m); w_flux = w_mag
        
        p['flux_weighted'], p['wsum_flux'], p['wsum_flux2'], p['flux_rmse'] = update_stats(p['flux_weighted'], p['wsum_flux'], p['wsum_flux2'], new_row['flux_raw'], w_flux)
        p['mag_weighted'], p['wsum_mag'], p['wsum_mag2'], p['mag_rmse'] = update_stats(p['mag_weighted'], p['wsum_mag'], p['wsum_mag2'], new_row['mag_raw'], w_mag)
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
