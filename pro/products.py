"""pro.products — the product registry + sidebar grouping."""
PRODUCTS = {
    # ---- velocity potential + wind (upper-level convergence/divergence) ----
    "vtp200": {"id": "vtp200", "title": "Velocity Potential & Wind Anomaly — 200 hPa",
               "name": "χ200 · Wind", "tag": "Upper",
               "desc": "200-hPa velocity-potential (divergence) and wind anomalies.",
               "kind": "vtp", "level": 200, "variables": ["uwnd", "vwnd"],
               "show_wind": True, "wind_scale": 50.0, "plot_scale": 1e-6,
               "vlim": 10.0, "cint": 2.5,
               "cb_label": "Velocity-Potential Anomaly  (1e6 m²s)"},
    "vtp500": {"id": "vtp500", "title": "Velocity Potential & Wind Anomaly — 500 hPa",
               "name": "χ500 · Wind", "tag": "Mid",
               "desc": "500-hPa velocity-potential and wind anomalies.",
               "kind": "vtp", "level": 500, "variables": ["uwnd", "vwnd"],
               "show_wind": True, "wind_scale": 50.0, "plot_scale": 1e-6,
               "vlim": 10.0, "cint": 2.5,
               "cb_label": "Velocity-Potential Anomaly  (1e6 m²s)"},
    "vtp850": {"id": "vtp850", "title": "Velocity Potential & Wind Anomaly — 850 hPa",
               "name": "χ850 · Wind", "tag": "Low",
               "desc": "850-hPa velocity-potential and wind anomalies.",
               "kind": "vtp", "level": 850, "variables": ["uwnd", "vwnd"],
               "show_wind": True, "wind_scale": 50.0, "plot_scale": 1e-6,
               "vlim": 10.0, "cint": 2.5,
               "cb_label": "Velocity-Potential Anomaly  (1e6 m²s)"},

    # ---- streamfunction + Rossby wave train ----
    "psi200": {"id": "psi200", "title": "Streamfunction Anomaly — 200 hPa",
               "name": "ψ200", "tag": "Upper",
               "desc": "200-hPa streamfunction anomaly (rotational circulation centers).",
               "kind": "psi", "level": 200, "variables": ["uwnd", "vwnd"],
               "show_wind": False, "plot_scale": 1e-6,
               "vlim": 40.0, "cint": 8.0,
               "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},
    "rwt200": {"id": "rwt200", "title": "Rossby Wave Train Circulation — 200 hPa",
               "name": "Wave Train ψ200", "tag": "Upper",
               "desc": "200-hPa streamfunction anomaly + wind: Rossby wave train "
                       "of alternating cyclonic/anticyclonic cells.",
               "kind": "psi", "level": 200, "variables": ["uwnd", "vwnd"],
               "show_wind": True, "wind_scale": 50.0, "plot_scale": 1e-6,
               "vlim": 40.0, "cint": 8.0,
               "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},

    # ---- geopotential height anomaly ----
    "hgt200": {"id": "hgt200", "title": "Geopotential Height Anomaly — 200 hPa",
               "name": "H200", "tag": "Upper",
               "desc": "200-hPa geopotential height anomaly (upper ridges & troughs).",
               "kind": "anom", "variable": "hgt", "level": 200,
               "show_wind": False, "plot_scale": 1.0,
               "vlim": 150.0, "cint": 30.0,
               "cb_label": "Geopotential Height Anomaly  (gpm)"},
    "hgt500": {"id": "hgt500", "title": "Geopotential Height Anomaly — 500 hPa",
               "name": "H500", "tag": "Mid",
               "desc": "500-hPa geopotential height anomaly (mid-tropospheric ridges & troughs).",
               "kind": "anom", "variable": "hgt", "level": 500,
               "show_wind": False, "plot_scale": 1.0,
               "vlim": 150.0, "cint": 30.0,
               "cb_label": "Geopotential Height Anomaly  (gpm)"},
    "hgt850": {"id": "hgt850", "title": "Geopotential Height Anomaly — 850 hPa",
               "name": "H850", "tag": "Low",
               "desc": "850-hPa geopotential height anomaly (low-level ridges & troughs).",
               "kind": "anom", "variable": "hgt", "level": 850,
               "show_wind": False, "plot_scale": 1.0,
               "vlim": 150.0, "cint": 30.0,
               "cb_label": "Geopotential Height Anomaly  (gpm)"},

    # ---- zonal wind anomaly ----
    "u200": {"id": "u200", "title": "Zonal Wind Anomaly — 200 hPa",
             "name": "U200", "tag": "Upper",
             "desc": "200-hPa zonal (east-west) wind anomaly.",
             "kind": "anom", "variable": "uwnd", "level": 200,
             "show_wind": False, "plot_scale": 1.0,
             "vlim": 15.0, "cint": 3.0,
             "cb_label": "Zonal Wind Anomaly  (m/s)"},
    "u850": {"id": "u850", "title": "Zonal Wind Anomaly — 850 hPa",
             "name": "U850", "tag": "Low",
             "desc": "850-hPa zonal (east-west) wind anomaly.",
             "kind": "anom", "variable": "uwnd", "level": 850,
             "show_wind": False, "plot_scale": 1.0,
             "vlim": 10.0, "cint": 2.0,
             "cb_label": "Zonal Wind Anomaly  (m/s)"},

    # ---- temperature anomaly ----
    "t200": {"id": "t200", "title": "Temperature Anomaly — 200 hPa",
             "name": "T200", "tag": "Upper",
             "desc": "200-hPa temperature anomaly.",
             "kind": "anom", "variable": "air", "level": 200,
             "show_wind": False, "plot_scale": 1.0,
             "vlim": 6.0, "cint": 1.5,
             "cb_label": "Temperature Anomaly  (K)"},
    "t850": {"id": "t850", "title": "Temperature Anomaly — 850 hPa",
             "name": "T850", "tag": "Low",
             "desc": "850-hPa temperature anomaly.",
             "kind": "anom", "variable": "air", "level": 850,
             "show_wind": False, "plot_scale": 1.0,
             "vlim": 8.0, "cint": 2.0,
             "cb_label": "Temperature Anomaly  (K)"},

    # ---- angular momentum budget: frictional torque (mountain torque needs
    #      orography, which this dataset does not carry — see README) ----
    "frict": {"id": "frict", "title": "Frictional Torque — Zonal (τx)",
              "name": "Friction τx", "tag": "Torque",
              "desc": "Surface zonal wind-stress anomaly (the zonal frictional-"
                      "torque driver) with the full surface stress vector, from "
                      "10-m winds via the bulk drag law.",
              "kind": "ft", "level": None, "variables": [],
              "comp": "x", "show_wind": True, "wind_scale": 55.0,
              "vec_scale": 100.0, "vec_step": 5, "vec_min": 10.0,
              "plot_scale": 100.0,
              "vlim": 30.0, "cint": 6.0,
              "cb_label": "Surface Zonal Stress Anomaly  (×10⁻² N/m²)"},
    "frict_y": {"id": "frict_y", "title": "Frictional Torque — Meridional (τy)",
                "name": "Friction τy", "tag": "Torque",
                "desc": "Surface meridional wind-stress anomaly (the meridional "
                        "frictional-torque driver) with the full stress vector.",
                "kind": "ft", "level": None, "variables": [],
                "comp": "y", "show_wind": True, "wind_scale": 55.0,
                "vec_scale": 100.0, "vec_step": 5, "vec_min": 10.0,
                "plot_scale": 100.0,
                "vlim": 18.0, "cint": 3.0,
                "cb_label": "Surface Meridional Stress Anomaly  (×10⁻² N/m²)"},
    "sstress": {"id": "sstress", "title": "Surface Wind Stress Magnitude (|τ|)",
                "name": "Stress |τ|", "tag": "Torque",
                "desc": "Magnitude of the surface wind-stress anomaly with the "
                        "stress vector — the full frictional-forcing field.",
                "kind": "ft", "level": None, "variables": [],
                "comp": "mag", "show_wind": True, "wind_scale": 55.0,
                "vec_scale": 100.0, "vec_step": 5, "vec_min": 10.0,
                "plot_scale": 100.0, "one_sided": True,
                "vlim": 30.0, "cint": 6.0,
                "cb_label": "Surface Stress Magnitude Anomaly  (×10⁻² N/m²)"},

    # ---- meridional wind anomaly (V) ----
    "v200": {"id": "v200", "title": "Meridional Wind Anomaly — 200 hPa",
             "name": "V200", "tag": "Upper",
             "desc": "200-hPa meridional (south-north) wind anomaly.",
             "kind": "anom", "variable": "vwnd", "level": 200,
             "show_wind": False, "plot_scale": 1.0,
             "vlim": 25.0, "cint": 5.0,
             "cb_label": "Meridional Wind Anomaly  (m/s)"},
    "v850": {"id": "v850", "title": "Meridional Wind Anomaly — 850 hPa",
             "name": "V850", "tag": "Low",
             "desc": "850-hPa meridional (south-north) wind anomaly.",
             "kind": "anom", "variable": "vwnd", "level": 850,
             "show_wind": False, "plot_scale": 1.0,
             "vlim": 15.0, "cint": 3.0,
             "cb_label": "Meridional Wind Anomaly  (m/s)"},

    # ---- relative humidity anomaly ----
    "rh850": {"id": "rh850", "title": "Relative Humidity Anomaly — 850 hPa",
              "name": "RH850", "tag": "Low",
              "desc": "850-hPa relative humidity anomaly.",
              "kind": "anom", "variable": "rhum", "level": 850,
              "show_wind": False, "plot_scale": 1.0,
              "vlim": 40.0, "cint": 8.0,
              "cb_label": "Relative Humidity Anomaly  (%)"},
    "rh700": {"id": "rh700", "title": "Relative Humidity Anomaly — 700 hPa",
              "name": "RH700", "tag": "Mid",
              "desc": "700-hPa relative humidity anomaly.",
              "kind": "anom", "variable": "rhum", "level": 700,
              "show_wind": False, "plot_scale": 1.0,
              "vlim": 40.0, "cint": 8.0,
              "cb_label": "Relative Humidity Anomaly  (%)"},
    "rh500": {"id": "rh500", "title": "Relative Humidity Anomaly — 500 hPa",
              "name": "RH500", "tag": "Mid",
              "desc": "500-hPa relative humidity anomaly.",
              "kind": "anom", "variable": "rhum", "level": 500,
              "show_wind": False, "plot_scale": 1.0,
              "vlim": 40.0, "cint": 8.0,
              "cb_label": "Relative Humidity Anomaly  (%)"},

    # ---- sea-level pressure anomaly ----
    "slp": {"id": "slp", "title": "Sea-Level Pressure Anomaly",
            "name": "SLP", "tag": "Surface",
            "desc": "MSLP anomaly — the classic surface pressure chart.",
            "kind": "anom", "variable": "slp", "level": None,
            "show_wind": False, "plot_scale": 1.0,
            "vlim": 35.0, "cint": 7.0,
            "cb_label": "Sea-Level Pressure Anomaly  (hPa)"},
    "srfp": {"id": "srfp", "title": "Surface Pressure Anomaly (ps)",
             "name": "ps", "tag": "Surface",
             "desc": "Daily surface-pressure anomaly — the real terrain-influenced "
                     "surface pressure field (not reduced to sea level).",
             "kind": "anom", "variable": "srfp", "level": None,
             "show_wind": False, "plot_scale": 1.0,
             "vlim": 25.0, "cint": 5.0,
             "cb_label": "Surface Pressure Anomaly  (hPa)"},

    # ---- streamfunction + wave train at 500 & 850 ----
    "psi500": {"id": "psi500", "title": "Streamfunction Anomaly — 500 hPa",
               "name": "ψ500", "tag": "Mid",
               "desc": "500-hPa streamfunction anomaly (mid-tropospheric circulation centers).",
               "kind": "psi", "level": 500, "variables": ["uwnd", "vwnd"],
               "show_wind": False, "plot_scale": 1e-6,
               "vlim": 40.0, "cint": 8.0,
               "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},
    "psi850": {"id": "psi850", "title": "Streamfunction Anomaly — 850 hPa",
               "name": "ψ850", "tag": "Low",
               "desc": "850-hPa streamfunction anomaly (low-level circulation centers).",
               "kind": "psi", "level": 850, "variables": ["uwnd", "vwnd"],
               "show_wind": False, "plot_scale": 1e-6,
               "vlim": 40.0, "cint": 8.0,
               "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},
    "rwt500": {"id": "rwt500", "title": "Rossby Wave Train Circulation — 500 hPa",
               "name": "Wave Train ψ500", "tag": "Mid",
               "desc": "500-hPa streamfunction anomaly + wind: mid-level Rossby wave train.",
               "kind": "psi", "level": 500, "variables": ["uwnd", "vwnd"],
               "show_wind": True, "wind_scale": 50.0, "plot_scale": 1e-6,
               "vlim": 40.0, "cint": 8.0,
               "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},

    # ---- advanced diagnostics ----
    "waf200": {"id": "waf200",
               "title": "Wave Flux — 200 hPa",
               "name": "Wave Flux 200", "tag": "Advanced",
               "desc": "Takaya–Nakamura wave-activity flux vectors over 200-hPa "
                       "streamfunction anomaly (Rossby wave propagation source/sink).",
               "kind": "waf", "level": 200, "variables": ["uwnd", "vwnd"],
               "show_wind": True, "wind_scale": 400.0, "plot_scale": 1e-6,
               "vec_scale": 1e-4, "vec_ref": 50.0, "vec_unit": "5×10⁵ m²/s²",
               "vec_step": 5, "vec_min": 15.0,
               "vlim": 40.0, "cint": 8.0,
               "cb_label": "Streamfunction Anomaly  (1e6 m²s)"},
    "rossby200": {"id": "rossby200",
                  "title": "Rossby-Wave Source, Train & Waveguide — 200 hPa",
                  "name": "Rossby Wave Composite", "tag": "Advanced",
                  "desc": "χ200′ source shading (green = upper divergence / "
                          "enhanced convection, brown = suppressed), ψ200′ wave-train "
                          "contours, Takaya–Nakamura wave-activity-flux arrows, and "
                          "the Ks≥5 stationary-wavenumber waveguide band.",
                  "kind": "rossby", "level": 200, "variables": ["uwnd", "vwnd"],
                  "plot_scale": 1e-6, "vlim": 10.0, "cint": 2.5,
                  "psi_interval": 6e6, "psi_vlim": 42e6,
                  "ks_threshold": 5.0, "ubar_min": 4.0,
                  "wind_scale": 400.0, "vec_scale": 1e-4,
                  "vec_ref": 50.0, "vec_unit": "5×10⁵ m²/s²",
                  "vec_step": 5, "vec_min": 15.0,
                  "cb_label": "Velocity-Potential Anomaly χ′  (1e6 m²s)"},
    "qgpv200": {"id": "qgpv200",
                "title": "QG Potential Vorticity Anomaly — 200 hPa",
                "name": "QG PV 200", "tag": "Advanced",
                "desc": "Quasi-geostrophic potential-vorticity anomaly at 200 hPa "
                        "(jet & wave-breaking diagnostics).",
                "kind": "qgpv", "level": 200, "variables": ["uwnd", "vwnd", "air"],
                "show_wind": False, "plot_scale": 1e6,
                "vlim": 320.0, "cint": 40.0,
                "cb_label": "QG PV Anomaly  (×10⁻⁶ s⁻¹)"},
    "eddy_vt": {"id": "eddy_vt",
                "title": "Eddy Meridional Flux v′T′ — 200 hPa",
                "name": "Eddy v′T′ 200", "tag": "Advanced",
                "desc": "Transient-eddy meridional heat flux v′T′ (deviation from "
                        "zonal mean of the anomaly) at 200 hPa.",
                "kind": "eddy", "level": 200, "variables": ["uwnd", "vwnd", "air"],
                "flux": "vt", "show_wind": False, "plot_scale": 1e-2,
                "vlim": 1.5, "cint": 0.25,
                "cb_label": "Eddy v′T′ Anomaly  (×10⁻² m s⁻¹ K)"},
    "eddy_uv": {"id": "eddy_uv",
                "title": "Eddy Momentum Flux u′v′ — 200 hPa",
                "name": "Eddy u′v′ 200", "tag": "Advanced",
                "desc": "Transient-eddy meridional momentum flux u′v′ at 200 hPa.",
                "kind": "eddy", "level": 200, "variables": ["uwnd", "vwnd", "air"],
                "flux": "uv", "show_wind": False, "plot_scale": 1e-2,
                "vlim": 4.0, "cint": 0.5,
                "cb_label": "Eddy u′v′  (×10⁻² m²/s²)"},
    "eady": {"id": "eady",
             "title": "Eady Baroclinic Growth Rate",
             "name": "Eady σ 850–500", "tag": "Advanced",
             "desc": "Eady baroclinic growth rate (850–500 hPa shear × static "
                     "stability); the 'storm-fuelling' instability index.",
             "kind": "eady", "level": 700, "variables": ["uwnd", "air"],
             "p_low": 850, "p_high": 500, "show_wind": False,
             "plot_scale": 1.0, "vlim": 1.2, "cint": 0.3,
             "cb_label": "Eady Growth Rate  (1/day)"},

    # ---- integrated water vapour transport ----
    "ivt": {"id": "ivt", "title": "Integrated Water Vapour Transport",
            "name": "IVT", "tag": "Moisture",
            "desc": "Column-integrated water-vapour transport |∫q·V dp| — the "
                    "atmospheric-river 'moisture highway' map.",
            "kind": "ivt", "level": None, "variables": ["uwnd", "vwnd", "air", "rhum"],
            "show_wind": True, "wind_scale": 1400.0, "plot_scale": 1.0,
            "vec_scale": 1.0, "vec_ref": 400.0, "vec_unit": "400 kg m⁻¹ s⁻¹",
            "vec_step": 5, "vec_min": 80.0,
            "one_sided": True,
            "vlim": 400.0, "cint": 50.0,
            "cb_label": "Integrated Water Vapour Transport  (kg m⁻¹ s⁻¹)"},

    # ---- QG omega forcing ----
    "qgforcing500": {"id": "qgforcing500", "title": "QG Omega Forcing — 500 hPa",
                     "name": "QG ω-forcing 500", "tag": "Dynamics",
                     "desc": "Quasi-geostrophic omega forcing −2∇·Q (Hoskins "
                             "Q-vector): red = forced ascent, blue = descent.",
                     "kind": "qgforcing", "level": 500,
                     "variables": ["uwnd", "vwnd", "air", "hgt"],
                     "min_lat": 12.0, "show_wind": False,
                     "plot_scale": 1e12,
                     "vlim": 4.0, "cint": 1.0,
                     "cb_label": "QG Omega Forcing  (×10⁻¹² K m⁻² s⁻¹)"},

    # ---- moist static energy anomaly ----
    "mse850": {"id": "mse850", "title": "Moist Static Energy Anomaly — 850 hPa",
               "name": "MSE 850", "tag": "Thermo",
               "desc": "Moist Static Energy (Cp·T + Lv·q + g·z) anomaly at 850 hPa "
                       "— boundary-layer convective/energetics field.",
               "kind": "mse", "level": 850, "variables": ["air", "rhum", "hgt"],
               "show_wind": False, "plot_scale": 1e-3, "invert_cbar": True,
               "vlim": 12.0, "cint": 3.0,
               "cb_label": "MSE Anomaly  (×10³ J/kg)"},
    "mse500": {"id": "mse500", "title": "Moist Static Energy Anomaly — 500 hPa",
               "name": "MSE 500", "tag": "Thermo",
               "desc": "Moist Static Energy (Cp·T + Lv·q + g·z) anomaly at 500 hPa.",
               "kind": "mse", "level": 500, "variables": ["air", "rhum", "hgt"],
               "show_wind": False, "plot_scale": 1e-3, "invert_cbar": True,
               "vlim": 9.0, "cint": 2.0,
               "cb_label": "MSE Anomaly  (×10³ J/kg)"},

    # ---- temperature advection ----
    "tadv850": {"id": "tadv850", "title": "Temperature Advection — 850 hPa",
                "name": "T-adv 850", "tag": "Dynamics",
                "desc": "−V·∇T at 850 hPa (warm advection red, cold advection blue) "
                        "in K/day — the classic frontal/isentropic forcing map.",
                "kind": "tadv", "level": 850, "variables": ["uwnd", "vwnd", "air"],
                "show_wind": False, "plot_scale": 86400.0,
                "vlim": 8.0, "cint": 2.0,
                "cb_label": "Temperature Advection  (K/day)"},

    # ---- geostrophic / ageostrophic wind ----
    "geowind300": {"id": "geowind300", "title": "Geostrophic Wind — 300 hPa",
                   "name": "Geo-wind 300", "tag": "Flow",
                   "desc": "Geostrophic wind speed from the height field with the "
                           "geostrophic vector (equator masked, f→0).",
                   "kind": "geowind", "level": 300, "variables": ["hgt"],
                   "min_lat": 12.0, "show_wind": True, "wind_scale": 45.0,
                   "vec_ref": 20.0, "vec_unit": "20 m/s", "plot_scale": 1.0,
                   "vec_step": 5, "vec_min": 12.0, "one_sided": True,
                   "vlim": 90.0, "cint": 15.0,
                   "cb_label": "Geostrophic Wind Speed  (m/s)"},
    "ageowind300": {"id": "ageowind300", "title": "Ageostrophic Wind — 300 hPa",
                    "name": "Ageo-wind 300", "tag": "Flow",
                    "desc": "Ageostrophic wind (V − Vg) magnitude & vector at 300 hPa "
                            "— the divergent/accelerating part of the flow.",
                    "kind": "ageowind", "level": 300, "variables": ["uwnd", "vwnd", "hgt"],
                    "min_lat": 12.0, "show_wind": True, "wind_scale": 20.0,
                    "vec_ref": 5.0, "vec_unit": "5 m/s", "plot_scale": 1.0,
                    "vec_step": 5, "vec_min": 2.5, "one_sided": True,
                    "vlim": 20.0, "cint": 4.0,
                    "cb_label": "Ageostrophic Wind Speed  (m/s)"},

    # ---- Hovmöller diagrams (daily, latitude-band averaged, longitude–time) ----
    "hov_u850": {"id": "hov_u850", "title": "Zonal Wind 850 hPa",
                 "name": "Hovmöller U850", "tag": "Hovmöller",
                 "desc": "Longitude–time Hovmöller of the 850-hPa zonal-wind "
                         "anomaly averaged 5°S–5°N (equatorial waves / MJO).",
                 "kind": "hov", "variable": "uwnd", "level": 850,
                 "lat_band": (-5, 5), "window": 120,
                 "plot_scale": 1.0,
                 "vlim": 6.0, "cint": 1.0,
                 "cb_label": "Zonal Wind Anomaly 850 hPa  (m/s)"},
    "hov_chi200": {"id": "hov_chi200", "title": "Velocity Potential 200 hPa",
                   "name": "Hovmöller χ200", "tag": "Hovmöller",
                   "desc": "Longitude–time Hovmöller of the 200-hPa velocity-"
                           "potential anomaly averaged 15°S–15°N (convection "
                           "propagation / MJO).",
                   "kind": "hov", "variable": "chi", "level": 200,
                   "lat_band": (-15, 15), "window": 120,
                   "plot_scale": 1e-6,
                   "vlim": 5.0, "cint": 1.0,
                   "cb_label": "Velocity-Potential Anomaly  (1e6 m²s)"},
}

# --- drop-in addons ---------------------------------------------------------
# Any product dicts (and custom kinds) defined in ./custom/*.py are merged into
# the registry here, so the sidebar /products list / generate all pick them up
# without touching a core file. See pro/addons.py and custom/_EXAMPLE.py.
from . import addons  # noqa: E402

PRODUCTS.update(addons.custom_products())


def custom_kind_configured(kind):
    """True if ``kind`` is a custom diagnostic contributed by an addon."""
    return kind in addons.custom_kinds()


GROUP_ORDER = ["Upper", "Mid", "Low", "Dynamics", "Thermo",
               "Moisture", "Torque", "Flow", "Advanced", "Surface",
               "Hovmöller", "Custom"]

def list_products():
    return [{"id": p["id"], "title": p["title"], "name": p["name"],
             "desc": p["desc"], "level": p["level"], "tag": p["tag"],
             "kind": p["kind"]}
            for p in PRODUCTS.values()]

def group_products():
    """Products grouped by tag, in GROUP_ORDER."""
    by_tag = {}
    for p in PRODUCTS.values():
        by_tag.setdefault(p["tag"], []).append(p)
    order = [t for t in GROUP_ORDER if t in by_tag] + \
            [t for t in by_tag if t not in GROUP_ORDER]
    out = []
    for tag in order:
        out.append({"tag": tag, "product_list": [
            {"id": p["id"], "title": p["title"], "name": p["name"],
             "desc": p["desc"], "level": p["level"], "tag": p["tag"],
             "kind": p["kind"]} for p in by_tag[tag]]})
    return out
