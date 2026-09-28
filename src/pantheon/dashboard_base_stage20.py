# ==========================================================
# STAGE 20 — FINAL INTERACTIVE 1982–2025 PANTHEON DASHBOARD
# ==========================================================
#
# Requires:
#
#   pantheon_2025
#   eligible_drivers
#   eligible_jaws_mat
#   rank_mat
#   alpha_post
#   idx_all
#   driver_years
#   actual_starts
#   networkx
#
# QoL additions:
#   - sticky section navigation
#   - fixed 4x2 desktop driver-summary card layout
#   - peak-season default in the driver explorer
#   - dynamically zoomed posterior-rank distribution
#   - time-aware 3D teammate network
#   - largest-component network view
#   - edge-filter-aware teammate side table
#   - labels in neighbourhood mode
#   - Teammate Chain game with daily/random challenges
#
# Output:
#
#   f1_pantheon_1982_2025_FINAL.html
#
# ==========================================================

import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError as exc:
    raise ImportError(
        "Stage 20 requires networkx for the interactive teammate network. "
        "Install it with: pip install networkx"
    ) from exc

# ==========================================================
# 0. REQUIRE FINAL OBJECTS
# ==========================================================

required_objects = [
    "pantheon_2025",
    "eligible_drivers",
    "eligible_jaws_mat",
    "rank_mat",
    "alpha_post",
    "idx_all",
    "driver_years",
    "actual_starts",
]


missing = [name for name in required_objects if name not in globals()]


if missing:

    raise RuntimeError("Missing required final objects: " + ", ".join(missing))


# ==========================================================
# 1. FINAL SETTINGS
# ==========================================================

START_YEAR = 1982

END_YEAR = int(CFG.ceiling_year)


assert END_YEAR == 2025


OUTPUT_DIR = Path(CFG.output_dir)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PANTHEON_DASHBOARD_PATH = OUTPUT_DIR / (
    f"f1_pantheon_" f"{START_YEAR}_{END_YEAR}_" f"FINAL.html"
)


OPEN_AFTER_BUILD = True


# ==========================================================
# 2. DISPLAY NAMES
# ==========================================================

if (
    "driverRef" in raw.drivers.columns
    and "forename" in raw.drivers.columns
    and "surname" in raw.drivers.columns
):

    display_name_lookup = {
        str(row.driverRef): (f"{row.forename} " f"{row.surname}").strip()
        for row in raw.drivers.itertuples(index=False)
    }

else:

    display_name_lookup = {}


def display_name(
    driver,
):

    return display_name_lookup.get(
        str(driver),
        str(driver)
        .replace(
            "_",
            " ",
        )
        .title(),
    )


# ==========================================================
# 3. SEASON-DETAIL SUPPORT
#
# The dashboard lets the user click a season in a driver's
# trajectory and inspect:
#   - the driver's latent season ability alpha;
#   - the constructor/car season effect beta;
#   - within-season driver and constructor ranks;
#   - teammate alpha comparisons;
#   - modern direct pace / qualifying contrast evidence.
# ==========================================================

if "posterior_final" in globals():

    _posterior_for_dashboard = posterior_final

else:

    _posterior_for_dashboard = final_trace_2025["posterior"].to_dataset()


car_post = np.asarray(
    _posterior_for_dashboard["car_csy"].values,
    dtype=float,
).reshape(
    -1,
    len(idx_all.active_constructor_years),
)


if car_post.shape[0] != alpha_post.shape[0]:

    raise RuntimeError(
        "Driver and constructor posterior draw counts do not match."
    )


# ----------------------------------------------------------
# Constructor display names
# ----------------------------------------------------------

constructor_name_lookup = {}


constructors_table = getattr(
    raw,
    "constructors",
    None,
)


if (
    constructors_table is not None
    and len(constructors_table)
):

    for _, constructor_row in constructors_table.iterrows():

        constructor_id = int(
            constructor_row["constructorId"]
        )

        constructor_name = None

        if (
            "name" in constructors_table.columns
            and pd.notna(
                constructor_row.get(
                    "name"
                )
            )
        ):

            constructor_name = str(
                constructor_row["name"]
            ).strip()

        if not constructor_name:

            constructor_name = str(
                constructor_row.get(
                    "constructorRef",
                    constructor_id,
                )
            ).replace(
                "_",
                " ",
            ).title()

        constructor_name_lookup[
            constructor_id
        ] = constructor_name


def constructor_name(
    constructor_id,
):

    return constructor_name_lookup.get(
        int(
            constructor_id
        ),
        f"Constructor {int(constructor_id)}",
    )


# ----------------------------------------------------------
# Draw-wise within-season ranks
# ----------------------------------------------------------

driver_season_rank_lookup = {}


for year in sorted(
    {
        int(y)
        for _, y
        in idx_all.active_driver_years
    }
):

    state_rows = [
        (
            driver,
            idx_all.dy_map[
                (
                    driver,
                    year,
                )
            ],
        )
        for driver, y
        in idx_all.active_driver_years
        if int(y) == year
    ]

    ids = np.array(
        [
            state_id
            for _, state_id
            in state_rows
        ],
        dtype=int,
    )

    values = alpha_post[
        :,
        ids,
    ]

    order = np.argsort(
        -values,
        axis=1,
    )

    ranks = np.empty_like(
        order,
        dtype=int,
    )

    rows = np.arange(
        values.shape[0]
    )[
        :,
        None
    ]

    ranks[
        rows,
        order
    ] = np.arange(
        1,
        len(ids) + 1,
    )[
        None,
        :
    ]

    for local_j, (
        driver,
        _,
    ) in enumerate(
        state_rows
    ):

        draws = ranks[
            :,
            local_j,
        ]

        driver_season_rank_lookup[
            (
                driver,
                year,
            )
        ] = {
            "mean":
                float(
                    draws.mean()
                ),
            "median":
                float(
                    np.median(
                        draws
                    )
                ),
            "lb":
                float(
                    np.quantile(
                        draws,
                        0.055,
                    )
                ),
            "ub":
                float(
                    np.quantile(
                        draws,
                        0.945,
                    )
                ),
            "field_size":
                int(
                    len(ids)
                ),
        }


constructor_season_rank_lookup = {}


for year in sorted(
    {
        int(y)
        for _, y
        in idx_all.active_constructor_years
    }
):

    state_rows = [
        (
            int(constructor_id),
            idx_all.csy_map[
                (
                    int(constructor_id),
                    year,
                )
            ],
        )
        for constructor_id, y
        in idx_all.active_constructor_years
        if int(y) == year
    ]

    ids = np.array(
        [
            state_id
            for _, state_id
            in state_rows
        ],
        dtype=int,
    )

    values = car_post[
        :,
        ids,
    ]

    order = np.argsort(
        -values,
        axis=1,
    )

    ranks = np.empty_like(
        order,
        dtype=int,
    )

    rows = np.arange(
        values.shape[0]
    )[
        :,
        None
    ]

    ranks[
        rows,
        order
    ] = np.arange(
        1,
        len(ids) + 1,
    )[
        None,
        :
    ]

    for local_j, (
        constructor_id,
        _,
    ) in enumerate(
        state_rows
    ):

        draws = ranks[
            :,
            local_j,
        ]

        constructor_season_rank_lookup[
            (
                constructor_id,
                year,
            )
        ] = {
            "mean":
                float(
                    draws.mean()
                ),
            "median":
                float(
                    np.median(
                        draws
                    )
                ),
            "lb":
                float(
                    np.quantile(
                        draws,
                        0.055,
                    )
                ),
            "ub":
                float(
                    np.quantile(
                        draws,
                        0.945,
                    )
                ),
            "field_size":
                int(
                    len(ids)
                ),
        }


# ----------------------------------------------------------
# Direct modern teammate contrast summaries.
#
# Positive values favour the selected driver.
# These are descriptive summaries of the observations used
# by the modern contrast likelihood, not extra model terms.
# ----------------------------------------------------------

def modern_contrast_summary(
    table,
    driver,
    teammate,
    year,
    constructor_id,
):

    if (
        table is None
        or len(
            table
        )
        == 0
    ):

        return None

    sub = table[
        (
            table["year"].astype(int)
            == int(year)
        )
        &
        (
            table["constructorId"].astype(int)
            == int(constructor_id)
        )
        &
        (
            (
                (
                    table["driver_a"]
                    == driver
                )
                &
                (
                    table["driver_b"]
                    == teammate
                )
            )
            |
            (
                (
                    table["driver_a"]
                    == teammate
                )
                &
                (
                    table["driver_b"]
                    == driver
                )
            )
        )
    ].copy()

    if sub.empty:

        return None

    oriented = np.where(
        sub[
            "driver_a"
        ].to_numpy()
        == driver,
        sub[
            "y"
        ].to_numpy(
            dtype=float
        ),
        -sub[
            "y"
        ].to_numpy(
            dtype=float
        ),
    )

    weights = sub[
        "weight"
    ].to_numpy(
        dtype=float
    )

    return {
        "mean":
            float(
                np.average(
                    oriented,
                    weights=weights,
                )
            ),
        "n_observations":
            int(
                len(
                    sub
                )
            ),
        "n_weekends":
            int(
                sub[
                    "raceId"
                ].nunique()
            ),
    }


_pace_contrast_for_dashboard = globals().get(
    "pace_contrast_all",
    globals().get(
        "pace_contrast",
        None,
    ),
)


_q_contrast_for_dashboard = globals().get(
    "q_contrast_all",
    globals().get(
        "q_contrast",
        None,
    ),
)


# ==========================================================
# 4. DRIVER TRAJECTORY + CLICKABLE SEASON PAYLOAD
# ==========================================================

trajectory_payload = {}

season_context_payload = {}


for driver in eligible_drivers:

    years = driver_years[
        driver
    ]

    ids = np.array(
        [
            idx_all.dy_map[
                (
                    driver,
                    int(
                        year
                    ),
                )
            ]
            for year
            in years
        ],
        dtype=int,
    )

    states = alpha_post[
        :,
        ids,
    ]

    trajectory_payload[
        driver
    ] = {
        "years":
            [
                int(
                    year
                )
                for year
                in years
            ],
        "mean":
            states.mean(
                axis=0
            ).tolist(),
        "median":
            np.median(
                states,
                axis=0,
            ).tolist(),
        "lb":
            np.quantile(
                states,
                0.055,
                axis=0,
            ).tolist(),
        "ub":
            np.quantile(
                states,
                0.945,
                axis=0,
            ).tolist(),
        "p_gt_zero":
            np.mean(
                states > 0,
                axis=0,
            ).tolist(),
    }

    season_context_payload[
        driver
    ] = {}

    for local_j, year in enumerate(
        years
    ):

        year = int(
            year
        )

        driver_draws = states[
            :,
            local_j,
        ]

        season_rank = driver_season_rank_lookup[
            (
                driver,
                year,
            )
        ]

        driver_starts_year = actual_starts[
            (
                actual_starts[
                    "driver"
                ]
                == driver
            )
            &
            (
                actual_starts[
                    "year"
                ].astype(
                    int
                )
                == year
            )
        ].copy()

        team_details = []

        for constructor_id, driver_team_rows in (
            driver_starts_year.groupby(
                "constructorId"
            )
        ):

            constructor_id = int(
                constructor_id
            )

            driver_races = set(
                driver_team_rows[
                    "raceId"
                ].astype(
                    int
                )
            )

            csy_key = (
                constructor_id,
                year,
            )

            if csy_key in idx_all.csy_map:

                car_draws = car_post[
                    :,
                    idx_all.csy_map[
                        csy_key
                    ],
                ]

                car_summary = {
                    "mean":
                        float(
                            car_draws.mean()
                        ),
                    "lb":
                        float(
                            np.quantile(
                                car_draws,
                                0.055,
                            )
                        ),
                    "ub":
                        float(
                            np.quantile(
                                car_draws,
                                0.945,
                            )
                        ),
                    "rank":
                        constructor_season_rank_lookup.get(
                            csy_key
                        ),
                }

            else:

                car_summary = None

            teammate_rows = actual_starts[
                (
                    actual_starts[
                        "constructorId"
                    ].astype(
                        int
                    )
                    == constructor_id
                )
                &
                (
                    actual_starts[
                        "year"
                    ].astype(
                        int
                    )
                    == year
                )
                &
                (
                    actual_starts[
                        "raceId"
                    ].astype(
                        int
                    )
                    .isin(
                        driver_races
                    )
                )
                &
                (
                    actual_starts[
                        "driver"
                    ]
                    != driver
                )
            ].copy()

            teammate_details = []

            for teammate, shared_rows in (
                teammate_rows.groupby(
                    "driver"
                )
            ):

                shared_starts = int(
                    shared_rows[
                        "raceId"
                    ].nunique()
                )

                teammate_key = (
                    teammate,
                    year,
                )

                if teammate_key in idx_all.dy_map:

                    teammate_draws = alpha_post[
                        :,
                        idx_all.dy_map[
                            teammate_key
                        ],
                    ]

                    delta_draws = (
                        driver_draws
                        -
                        teammate_draws
                    )

                    teammate_rank = (
                        driver_season_rank_lookup.get(
                            teammate_key
                        )
                    )

                    teammate_summary = {
                        "alpha_mean":
                            float(
                                teammate_draws.mean()
                            ),
                        "alpha_lb":
                            float(
                                np.quantile(
                                    teammate_draws,
                                    0.055,
                                )
                            ),
                        "alpha_ub":
                            float(
                                np.quantile(
                                    teammate_draws,
                                    0.945,
                                )
                            ),
                        "rank":
                            teammate_rank,
                        "delta_mean":
                            float(
                                delta_draws.mean()
                            ),
                        "delta_lb":
                            float(
                                np.quantile(
                                    delta_draws,
                                    0.055,
                                )
                            ),
                        "delta_ub":
                            float(
                                np.quantile(
                                    delta_draws,
                                    0.945,
                                )
                            ),
                        "p_driver_better":
                            float(
                                np.mean(
                                    delta_draws > 0
                                )
                            ),
                    }

                else:

                    teammate_summary = {
                        "alpha_mean":
                            None,
                        "alpha_lb":
                            None,
                        "alpha_ub":
                            None,
                        "rank":
                            None,
                        "delta_mean":
                            None,
                        "delta_lb":
                            None,
                        "delta_ub":
                            None,
                        "p_driver_better":
                            None,
                    }

                pace_evidence = (
                    modern_contrast_summary(
                        _pace_contrast_for_dashboard,
                        driver,
                        teammate,
                        year,
                        constructor_id,
                    )
                    if year
                    >= int(
                        CFG.modern_floor_year
                    )
                    else None
                )

                quali_evidence = (
                    modern_contrast_summary(
                        _q_contrast_for_dashboard,
                        driver,
                        teammate,
                        year,
                        constructor_id,
                    )
                    if year
                    >= int(
                        CFG.modern_floor_year
                    )
                    else None
                )

                teammate_details.append(
                    {
                        "driver":
                            teammate,
                        "display_name":
                            display_name(
                                teammate
                            ),
                        "shared_starts":
                            shared_starts,
                        **teammate_summary,
                        "pace_evidence":
                            pace_evidence,
                        "quali_evidence":
                            quali_evidence,
                    }
                )

            teammate_details = sorted(
                teammate_details,
                key=lambda row: (
                    -row[
                        "shared_starts"
                    ],
                    row[
                        "display_name"
                    ],
                ),
            )

            team_details.append(
                {
                    "constructor_id":
                        constructor_id,
                    "constructor_name":
                        constructor_name(
                            constructor_id
                        ),
                    "starts":
                        int(
                            len(
                                driver_races
                            )
                        ),
                    "car":
                        car_summary,
                    "teammates":
                        teammate_details,
                }
            )

        team_details = sorted(
            team_details,
            key=lambda row: (
                -row[
                    "starts"
                ],
                row[
                    "constructor_name"
                ],
            ),
        )

        season_context_payload[
            driver
        ][
            str(
                year
            )
        ] = {
            "year":
                year,
            "era":
                (
                    "Modern pace + timed qualifying"
                    if year
                    >= int(
                        CFG.modern_floor_year
                    )
                    else
                    "Historical ordinal qualifying/race"
                ),
            "driver_alpha": {
                "mean":
                    float(
                        driver_draws.mean()
                    ),
                "lb":
                    float(
                        np.quantile(
                            driver_draws,
                            0.055,
                        )
                    ),
                "ub":
                    float(
                        np.quantile(
                            driver_draws,
                            0.945,
                        )
                    ),
                "p_gt_zero":
                    float(
                        np.mean(
                            driver_draws > 0
                        )
                    ),
            },
            "driver_rank":
                season_rank,
            "teams":
                team_details,
        }


# ==========================================================
# 5. ENTIRE MODELLED TEAMMATE NETWORK — 3D PAYLOAD
# ==========================================================
#
# The web page contains a fully interactive 3D teammate
# network covering every driver represented in the model.
#
# Node encoding:
#   colour = mean alpha across the driver's modelled seasons
#   size   = number of actual race starts in 1982–2025
#
# Edge encoding:
#   connection = at least one shared actual race start for the
#                same constructor
#   strength   = number of shared actual race starts
#
# The 3D coordinates are a force-directed layout only. They
# have no physical/statistical axis interpretation; distance
# is used only to make network topology readable.
# ==========================================================

# ----------------------------------------------------------
# Modelled seasons by driver
# ----------------------------------------------------------

_network_years_by_driver = {}

for _driver, _year in idx_all.active_driver_years:

    _network_years_by_driver.setdefault(
        str(_driver),
        [],
    ).append(
        int(_year)
    )


for _driver in _network_years_by_driver:

    _network_years_by_driver[_driver] = sorted(
        set(
            _network_years_by_driver[_driver]
        )
    )


_network_drivers = sorted(
    _network_years_by_driver
)


_network_driver_set = set(
    _network_drivers
)


# ----------------------------------------------------------
# Actual-start counts
# ----------------------------------------------------------

_network_start_counts = (
    actual_starts.loc[
        actual_starts[
            "driver"
        ].isin(
            _network_driver_set
        )
    ]
    .groupby(
        "driver"
    )["raceId"]
    .nunique()
    .to_dict()
)


# ----------------------------------------------------------
# Build actual-start teammate edges
# ----------------------------------------------------------

_network_edge_acc = {}


for (
    _race_id,
    _year,
    _constructor_id,
), _grp in actual_starts.groupby(
    [
        "raceId",
        "year",
        "constructorId",
    ]
):

    _drivers_here = sorted(
        set(
            _grp.loc[
                _grp[
                    "driver"
                ].isin(
                    _network_driver_set
                ),
                "driver",
            ]
            .dropna()
            .astype(
                str
            )
        )
    )

    if len(
        _drivers_here
    ) < 2:

        continue

    for _i in range(
        len(
            _drivers_here
        ) - 1
    ):

        for _j in range(
            _i + 1,
            len(
                _drivers_here
            )
        ):

            _a = _drivers_here[
                _i
            ]

            _b = _drivers_here[
                _j
            ]

            _key = (
                _a,
                _b,
            )

            if _key not in _network_edge_acc:

                _network_edge_acc[
                    _key
                ] = {
                    "shared_starts": 0,
                    "years": {},
                    "constructors": {},
                }

            _rec = _network_edge_acc[
                _key
            ]

            _rec[
                "shared_starts"
            ] += 1

            _rec[
                "years"
            ][
                int(
                    _year
                )
            ] = (
                _rec[
                    "years"
                ].get(
                    int(
                        _year
                    ),
                    0,
                )
                + 1
            )

            _rec[
                "constructors"
            ][
                int(
                    _constructor_id
                )
            ] = (
                _rec[
                    "constructors"
                ].get(
                    int(
                        _constructor_id
                    ),
                    0,
                )
                + 1
            )


# ----------------------------------------------------------
# Edge posterior summaries
# ----------------------------------------------------------

_network_edges = []

_network_teammates = {
    driver: set()
    for driver in _network_drivers
}


for (
    _a,
    _b,
), _rec in _network_edge_acc.items():

    _shared_years = sorted(
        _rec[
            "years"
        ]
    )

    _common_state_years = [
        year
        for year in _shared_years
        if (
            _a,
            int(
                year
            ),
        )
        in idx_all.dy_map
        and (
            _b,
            int(
                year
            ),
        )
        in idx_all.dy_map
    ]

    _delta_mean = None
    _delta_lb = None
    _delta_ub = None
    _p_a_better = None

    if _common_state_years:

        _delta_draws = np.column_stack(
            [
                alpha_post[
                    :,
                    idx_all.dy_map[
                        (
                            _a,
                            int(
                                year
                            ),
                        )
                    ],
                ]
                -
                alpha_post[
                    :,
                    idx_all.dy_map[
                        (
                            _b,
                            int(
                                year
                            ),
                        )
                    ],
                ]
                for year in _common_state_years
            ]
        ).mean(
            axis=1
        )

        _delta_mean = float(
            _delta_draws.mean()
        )

        _delta_lb = float(
            np.quantile(
                _delta_draws,
                0.055,
            )
        )

        _delta_ub = float(
            np.quantile(
                _delta_draws,
                0.945,
            )
        )

        _p_a_better = float(
            np.mean(
                _delta_draws
                >
                0
            )
        )

    _constructor_parts = []

    for (
        _constructor_id,
        _n,
    ) in sorted(
        _rec[
            "constructors"
        ].items(),
        key=lambda item:
            item[
                1
            ],
        reverse=True,
    ):

        _constructor_parts.append(
            f"{constructor_name(_constructor_id)} ({_n})"
        )

    _network_edges.append(
        {
            "a":
                _a,
            "b":
                _b,
            "shared_starts":
                int(
                    _rec[
                        "shared_starts"
                    ]
                ),
            "shared_seasons":
                [
                    int(
                        year
                    )
                    for year in _shared_years
                ],
            "first_year":
                int(
                    min(
                        _shared_years
                    )
                ),
            "last_year":
                int(
                    max(
                        _shared_years
                    )
                ),
            "year_counts":
                {
                    str(
                        int(
                            year
                        )
                    ):
                        int(
                            count
                        )
                    for year, count
                    in _rec[
                        "years"
                    ].items()
                },
            "constructors":
                _constructor_parts,
            "delta_a_minus_b_mean":
                _delta_mean,
            "delta_a_minus_b_lb":
                _delta_lb,
            "delta_a_minus_b_ub":
                _delta_ub,
            "p_a_better":
                _p_a_better,
        }
    )

    _network_teammates[
        _a
    ].add(
        _b
    )

    _network_teammates[
        _b
    ].add(
        _a
    )


_network_edges.sort(
    key=lambda edge:
        edge[
            "shared_starts"
        ],
    reverse=True,
)


# ----------------------------------------------------------
# Time-aware 3D network coordinates
#
# x/y: reproducible force-directed teammate topology
# z:   modelled-career midpoint year
# ----------------------------------------------------------

_network_graph = nx.Graph()


for _driver in _network_drivers:

    _network_graph.add_node(
        _driver
    )


for _edge in _network_edges:

    _network_graph.add_edge(
        _edge[
            "a"
        ],
        _edge[
            "b"
        ],
        weight=float(
            np.log1p(
                _edge[
                    "shared_starts"
                ]
            )
        ),
    )


_n_network_nodes = len(
    _network_graph
)


_network_k = (
    2.3
    /
    max(
        _n_network_nodes,
        1,
    ) ** 0.5
)


_network_pos_xy = nx.spring_layout(
    _network_graph,
    dim=2,
    seed=42,
    k=_network_k,
    iterations=600,
    weight="weight",
)


_network_xy_matrix = np.array(
    [
        _network_pos_xy[
            driver
        ]
        for driver in _network_drivers
    ],
    dtype=float,
)


if len(
    _network_xy_matrix
):

    _network_xy_matrix -= _network_xy_matrix.mean(
        axis=0,
        keepdims=True,
    )

    _network_scale = float(
        np.max(
            np.abs(
                _network_xy_matrix
            )
        )
    )

    if _network_scale > 0:

        _network_xy_matrix /= _network_scale


_network_pos = {}


for _i, _driver in enumerate(
    _network_drivers
):

    _years = _network_years_by_driver[
        _driver
    ]

    _career_midpoint_year = 0.5 * (
        float(
            min(
                _years
            )
        )
        +
        float(
            max(
                _years
            )
        )
    )

    _network_pos[
        _driver
    ] = np.array(
        [
            float(
                _network_xy_matrix[
                    _i,
                    0
                ]
            ),
            float(
                _network_xy_matrix[
                    _i,
                    1
                ]
            ),
            _career_midpoint_year,
        ],
        dtype=float,
    )


# ----------------------------------------------------------
# Node posterior / career summaries
# ----------------------------------------------------------

_pantheon_row_lookup = {
    str(
        row.driver
    ):
        row
    for row in pantheon_2025.itertuples(
        index=False
    )
}


_network_nodes = []


for _driver in _network_drivers:

    _years = _network_years_by_driver[
        _driver
    ]

    _ids = np.array(
        [
            idx_all.dy_map[
                (
                    _driver,
                    int(
                        year
                    ),
                )
            ]
            for year in _years
        ],
        dtype=int,
    )

    _career_draws = alpha_post[
        :,
        _ids,
    ].mean(
        axis=1
    )

    _pantheon_row = _pantheon_row_lookup.get(
        _driver
    )

    _coords = _network_pos[
        _driver
    ]

    _network_nodes.append(
        {
            "driver":
                _driver,
            "display_name":
                display_name(
                    _driver
                ),
            "x":
                float(
                    _coords[
                        0
                    ]
                ),
            "y":
                float(
                    _coords[
                        1
                    ]
                ),
            "z":
                float(
                    _coords[
                        2
                    ]
                ),
            "career_mean_alpha":
                float(
                    _career_draws.mean()
                ),
            "career_alpha_lb":
                float(
                    np.quantile(
                        _career_draws,
                        0.055,
                    )
                ),
            "career_alpha_ub":
                float(
                    np.quantile(
                        _career_draws,
                        0.945,
                    )
                ),
            "starts":
                int(
                    _network_start_counts.get(
                        _driver,
                        0,
                    )
                ),
            "modelled_seasons":
                int(
                    len(
                        _years
                    )
                ),
            "first_year":
                int(
                    min(
                        _years
                    )
                ),
            "last_year":
                int(
                    max(
                        _years
                    )
                ),
            "unique_teammates":
                int(
                    len(
                        _network_teammates[
                            _driver
                        ]
                    )
                ),
            "pantheon_rank":
                (
                    int(
                        _pantheon_row.rank
                    )
                    if _pantheon_row is not None
                    else None
                ),
            "jaws_mean":
                (
                    float(
                        _pantheon_row.jaws_mean
                    )
                    if _pantheon_row is not None
                    else None
                ),
        }
    )


_network_alpha_absmax = float(
    max(
        0.25,
        np.max(
            np.abs(
                [
                    node[
                        "career_mean_alpha"
                    ]
                    for node in _network_nodes
                ]
            )
        ),
    )
)


_network_components = list(
    nx.connected_components(
        _network_graph
    )
)


_network_largest_component = (
    max(
        (
            len(
                component
            )
            for component in _network_components
        ),
        default=0,
    )
)


network_payload = {
    "nodes":
        _network_nodes,
    "edges":
        _network_edges,
    "alpha_absmax":
        _network_alpha_absmax,
    "n_components":
        int(
            len(
                _network_components
            )
        ),
    "largest_component":
        int(
            _network_largest_component
        ),
}


# ==========================================================
# 6. TEAMMATE CHAIN GAME PAYLOAD
# ==========================================================
#
# The game uses the FULL actual-start teammate graph (1+ shared
# starts), irrespective of the visual network's current edge filter.
#
# Difficulty combines:
#   - shortest teammate-chain length;
#   - endpoint obscurity, using starts/wins/podiums as a transparent
#     recognition proxy rather than Pantheon ability;
#   - how historical the endpoints are;
#   - scarcity of equally-short routes.
#
# The final 0–100 score is a percentile within the generated challenge
# pool, so the Easy/Medium/Hard/Expert/Nightmare labels are calibrated
# to the actual 1982–2025 graph rather than arbitrary raw cutoffs.
# ==========================================================

from collections import deque


# ----------------------------------------------------------
# Recognition / fame proxy inputs
# ----------------------------------------------------------

_game_driver_id_to_ref = {}

if (
    "driverId" in raw.drivers.columns
    and "driverRef" in raw.drivers.columns
):

    _game_driver_id_to_ref = {
        int(row.driverId):
            str(row.driverRef)
        for row in raw.drivers.itertuples(
            index=False
        )
    }


_game_wins = {
    driver: 0
    for driver in _network_drivers
}

_game_podiums = {
    driver: 0
    for driver in _network_drivers
}


if (
    hasattr(
        raw,
        "results",
    )
    and hasattr(
        raw,
        "races",
    )
    and "driverId" in raw.results.columns
    and "raceId" in raw.results.columns
    and "positionOrder" in raw.results.columns
    and "raceId" in raw.races.columns
    and "year" in raw.races.columns
):

    _game_results = (
        raw.results[
            [
                "raceId",
                "driverId",
                "positionOrder",
            ]
        ]
        .merge(
            raw.races[
                [
                    "raceId",
                    "year",
                ]
            ],
            on=
                "raceId",
            how=
                "left",
            validate=
                "many_to_one",
        )
    )

    _game_results = _game_results[
        (
            _game_results[
                "year"
            ]
            >=
            START_YEAR
        )
        &
        (
            _game_results[
                "year"
            ]
            <=
            END_YEAR
        )
    ].copy()

    _game_results[
        "driver"
    ] = (
        _game_results[
            "driverId"
        ]
        .map(
            _game_driver_id_to_ref
        )
    )

    _game_results[
        "positionOrder"
    ] = pd.to_numeric(
        _game_results[
            "positionOrder"
        ],
        errors=
            "coerce",
    )

    _game_results = _game_results[
        _game_results[
            "driver"
        ].isin(
            _network_driver_set
        )
    ]

    _game_wins.update(
        _game_results[
            _game_results[
                "positionOrder"
            ]
            ==
            1
        ]
        .groupby(
            "driver"
        )
        .size()
        .astype(int)
        .to_dict()
    )

    _game_podiums.update(
        _game_results[
            _game_results[
                "positionOrder"
            ].isin(
                [
                    1,
                    2,
                    3,
                ]
            )
        ]
        .groupby(
            "driver"
        )
        .size()
        .astype(int)
        .to_dict()
    )


_game_max_starts = max(
    [
        int(
            _network_start_counts.get(
                driver,
                0,
            )
        )
        for driver in _network_drivers
    ]
    +
    [
        1
    ]
)

_game_max_wins = max(
    list(
        _game_wins.values()
    )
    +
    [
        1
    ]
)

_game_max_podiums = max(
    list(
        _game_podiums.values()
    )
    +
    [
        1
    ]
)


_game_driver_metrics = {}


for _driver in _network_drivers:

    _years = _network_years_by_driver[
        _driver
    ]

    _starts = int(
        _network_start_counts.get(
            _driver,
            0,
        )
    )

    _wins = int(
        _game_wins.get(
            _driver,
            0,
        )
    )

    _podiums = int(
        _game_podiums.get(
            _driver,
            0,
        )
    )

    _fame_score = (
        0.45
        *
        (
            np.log1p(
                _starts
            )
            /
            np.log1p(
                _game_max_starts
            )
        )
        +
        0.35
        *
        (
            np.log1p(
                _wins
            )
            /
            np.log1p(
                _game_max_wins
            )
        )
        +
        0.20
        *
        (
            np.log1p(
                _podiums
            )
            /
            np.log1p(
                _game_max_podiums
            )
        )
    )

    _first_year = int(
        min(
            _years
        )
    )

    _last_year = int(
        max(
            _years
        )
    )

    _midpoint = 0.5 * (
        _first_year
        +
        _last_year
    )

    _year_span = max(
        END_YEAR
        -
        START_YEAR,
        1,
    )

    # A driver gets a larger historical score if both their final
    # season and career midpoint are further from the modern endpoint.
    _historical_score = float(
        np.clip(
            0.60
            *
            (
                END_YEAR
                -
                _last_year
            )
            /
            _year_span
            +
            0.40
            *
            (
                END_YEAR
                -
                _midpoint
            )
            /
            _year_span,
            0.0,
            1.0,
        )
    )

    _game_driver_metrics[
        _driver
    ] = {
        "display_name":
            display_name(
                _driver
            ),
        "starts":
            _starts,
        "wins":
            _wins,
        "podiums":
            _podiums,
        "first_year":
            _first_year,
        "last_year":
            _last_year,
        "fame_score":
            float(
                _fame_score
            ),
        "obscurity_score":
            float(
                1.0
                -
                _fame_score
            ),
        "historical_score":
            _historical_score,
    }


# ----------------------------------------------------------
# Build challenge candidates from the largest connected component
# ----------------------------------------------------------

_game_components = list(
    nx.connected_components(
        _network_graph
    )
)

_game_component = (
    max(
        _game_components,
        key=
            len,
    )
    if _game_components
    else set()
)

_game_nodes = sorted(
    _game_component
)

_game_node_order = {
    driver: i
    for i, driver
    in enumerate(
        _game_nodes
    )
}


def _game_bfs_distances_and_counts(
    source,
):
    """
    Return shortest-path distance and number of shortest paths from
    `source` to every node in the unweighted teammate graph.
    """

    distances = {
        source: 0
    }

    path_counts = {
        source: 1
    }

    queue = deque(
        [
            source
        ]
    )

    while queue:

        current = queue.popleft()

        next_distance = (
            distances[
                current
            ]
            +
            1
        )

        for neighbour in _network_graph.neighbors(
            current
        ):

            if neighbour not in _game_component:

                continue

            if neighbour not in distances:

                distances[
                    neighbour
                ] = next_distance

                path_counts[
                    neighbour
                ] = path_counts[
                    current
                ]

                queue.append(
                    neighbour
                )

            elif (
                distances[
                    neighbour
                ]
                ==
                next_distance
            ):

                path_counts[
                    neighbour
                ] += path_counts[
                    current
                ]

    return (
        distances,
        path_counts,
    )


_game_candidate_rows = []


for _i, _source in enumerate(
    _game_nodes
):

    (
        _distances,
        _path_counts,
    ) = _game_bfs_distances_and_counts(
        _source
    )

    for _target in _game_nodes[
        _i
        +
        1:
    ]:

        if _target not in _distances:

            continue

        _distance = int(
            _distances[
                _target
            ]
        )

        # Direct teammates are too trivial for generated puzzles.
        # Extremely long chains become more frustrating than fun.
        if (
            _distance
            <
            2
            or
            _distance
            >
            7
        ):

            continue

        _n_shortest_paths = int(
            _path_counts[
                _target
            ]
        )

        _a_metric = _game_driver_metrics[
            _source
        ]

        _b_metric = _game_driver_metrics[
            _target
        ]

        _endpoint_obscurity = 0.5 * (
            _a_metric[
                "obscurity_score"
            ]
            +
            _b_metric[
                "obscurity_score"
            ]
        )

        _historical_difficulty = 0.5 * (
            _a_metric[
                "historical_score"
            ]
            +
            _b_metric[
                "historical_score"
            ]
        )

        # Distance 2 is deliberately easy; distance 6+ saturates the
        # graph-length contribution.
        _path_difficulty = float(
            np.clip(
                (
                    _distance
                    -
                    1
                )
                /
                5.0,
                0.0,
                1.0,
            )
        )

        # One unique shortest route is hardest. Several equally-short
        # routes make the puzzle more forgiving.
        _route_scarcity = float(
            1.0
            /
            (
                1.0
                +
                np.log2(
                    max(
                        _n_shortest_paths,
                        1,
                    )
                )
            )
        )

        _raw_difficulty = (
            0.50
            *
            _path_difficulty
            +
            0.25
            *
            _endpoint_obscurity
            +
            0.15
            *
            _historical_difficulty
            +
            0.10
            *
            _route_scarcity
        )

        _game_candidate_rows.append(
            {
                "a":
                    _source,
                "b":
                    _target,
                "optimal_edges":
                    _distance,
                "n_shortest_paths":
                    _n_shortest_paths,
                "path_difficulty":
                    _path_difficulty,
                "endpoint_obscurity":
                    float(
                        _endpoint_obscurity
                    ),
                "historical_difficulty":
                    float(
                        _historical_difficulty
                    ),
                "route_scarcity":
                    _route_scarcity,
                "raw_difficulty":
                    float(
                        _raw_difficulty
                    ),
            }
        )


_game_candidates = pd.DataFrame(
    _game_candidate_rows
)


if _game_candidates.empty:

    raise RuntimeError(
        "Could not construct any Teammate Chain challenge pairs."
    )


# Percentile calibration means difficulty tiers are balanced against the
# graph that actually exists rather than arbitrary score cut points.
_game_candidates[
    "difficulty_score"
] = (
    100.0
    *
    _game_candidates[
        "raw_difficulty"
    ]
    .rank(
        pct=True,
        method=
            "average",
    )
)


def _game_difficulty_tier(
    score,
):

    if score <= 20:

        return "Easy"

    if score <= 45:

        return "Medium"

    if score <= 70:

        return "Hard"

    if score <= 90:

        return "Expert"

    return "Nightmare"


_game_candidates[
    "difficulty_tier"
] = (
    _game_candidates[
        "difficulty_score"
    ]
    .map(
        _game_difficulty_tier
    )
)


# Keep the embedded HTML payload compact while retaining plenty of
# variety in every difficulty band.
_game_rng = np.random.default_rng(
    20250908
)

_game_bank_parts = []


for _tier in [
    "Easy",
    "Medium",
    "Hard",
    "Expert",
    "Nightmare",
]:

    _tier_frame = _game_candidates[
        _game_candidates[
            "difficulty_tier"
        ]
        ==
        _tier
    ]

    if len(
        _tier_frame
    ) > 300:

        _tier_frame = _tier_frame.sample(
            n=
                300,
            random_state=
                int(
                    _game_rng.integers(
                        0,
                        2**31
                        -
                        1,
                    )
                ),
        )

    _game_bank_parts.append(
        _tier_frame
    )


_game_bank = (
    pd.concat(
        _game_bank_parts,
        ignore_index=True,
    )
    .sort_values(
        [
            "difficulty_score",
            "a",
            "b",
        ]
    )
    .reset_index(
        drop=True
    )
)


teammate_game_payload = {
    "drivers":
        _game_driver_metrics,
    "challenges":
        [
            {
                "a":
                    str(
                        row.a
                    ),
                "b":
                    str(
                        row.b
                    ),
                "optimal_edges":
                    int(
                        row.optimal_edges
                    ),
                "n_shortest_paths":
                    int(
                        row.n_shortest_paths
                    ),
                "difficulty_score":
                    float(
                        row.difficulty_score
                    ),
                "difficulty_tier":
                    str(
                        row.difficulty_tier
                    ),
                "path_difficulty":
                    float(
                        row.path_difficulty
                    ),
                "endpoint_obscurity":
                    float(
                        row.endpoint_obscurity
                    ),
                "historical_difficulty":
                    float(
                        row.historical_difficulty
                    ),
                "route_scarcity":
                    float(
                        row.route_scarcity
                    ),
            }
            for row in _game_bank.itertuples(
                index=False
            )
        ],
    "metadata": {
        "graph_edge_threshold":
            1,
        "challenge_bank_size":
            int(
                len(
                    _game_bank
                )
            ),
        "candidate_pairs":
            int(
                len(
                    _game_candidates
                )
            ),
        "largest_component_drivers":
            int(
                len(
                    _game_component
                )
            ),
        "difficulty_formula": {
            "path_length":
                0.50,
            "endpoint_obscurity":
                0.25,
            "historical_distance":
                0.15,
            "route_scarcity":
                0.10,
        },
    },
}



# ==========================================================
# 6. PAIRWISE OFFICIAL JAWS MATRIX
# ==========================================================

n_drivers = len(eligible_drivers)


pairwise = np.zeros(
    (
        n_drivers,
        n_drivers,
    ),
    dtype=float,
)


for i in range(n_drivers):

    pairwise[i, :] = np.mean(
        eligible_jaws_mat[
            :,
            i,
            None,
        ]
        > eligible_jaws_mat,
        axis=0,
    )

    pairwise[i, i] = 0.5


# ==========================================================
# 7. POSTERIOR RANK DISTRIBUTIONS
# ==========================================================

rank_distribution = {}


for j, driver in enumerate(eligible_drivers):

    counts = np.bincount(
        rank_mat[
            :,
            j,
        ],
        minlength=n_drivers + 1,
    )[1:]

    probs = counts / counts.sum()

    rank_distribution[driver] = probs.tolist()


# ==========================================================
# 8. LEADERBOARD PAYLOAD
# ==========================================================

leaderboard_payload = []


for row in pantheon_2025.itertuples(
    index=False
):

    driver = row.driver

    leaderboard_payload.append(
        {
            "rank":
                int(
                    row.rank
                ),
            "driver":
                driver,
            "display_name":
                display_name(
                    driver
                ),
            "jaws_mean":
                float(
                    row.jaws_mean
                ),
            "jaws_median":
                float(
                    row.jaws_median
                ),
            "jaws_lb":
                float(
                    row.jaws_lb
                ),
            "jaws_ub":
                float(
                    row.jaws_ub
                ),
            "peak_alpha":
                float(
                    row.top5_peak_alpha_mean
                ),
            "career_surplus":
                float(
                    row.career_surplus_mean
                ),
            "peak_score":
                float(
                    row.peak_score_mean
                ),
            "longevity_score":
                float(
                    row.longevity_score_mean
                ),
            "rank_mean":
                float(
                    row.rank_mean
                ),
            "rank_median":
                float(
                    row.rank_median
                ),
            "rank_lb":
                float(
                    row.rank_lb
                ),
            "rank_ub":
                float(
                    row.rank_ub
                ),
            "rank_width":
                float(
                    row.rank_ub
                    -
                    row.rank_lb
                ),
            "p_rank_1":
                float(
                    row.p_rank_1
                ),
            "seasons":
                int(
                    row.seasons
                ),
            "starts":
                int(
                    row.starts
                ),
            "unique_teammates":
                int(
                    row.unique_teammates
                ),
        }
    )


# ==========================================================
# 9. PAYLOAD
# ==========================================================

payload = {
    "drivers":
        list(
            eligible_drivers
        ),
    "leaderboard":
        leaderboard_payload,
    "trajectories":
        trajectory_payload,
    "seasons":
        season_context_payload,
    "rank_distributions":
        rank_distribution,
    "pairwise":
        pairwise.tolist(),
    "network":
        network_payload,
    "game":
        teammate_game_payload,
    "metadata": {
        "n_drivers":
            int(
                len(
                    eligible_drivers
                )
            ),
        "n_draws":
            int(
                eligible_jaws_mat.shape[
                    0
                ]
            ),
        "start_year":
            START_YEAR,
        "end_year":
            END_YEAR,
        "peak_weight":
            0.60,
        "longevity_weight":
            0.40,
        "min_starts":
            35,
        "min_teammates":
            3,
        "min_seasons":
            5,
        "modern_floor_year":
            int(
                CFG.modern_floor_year
            ),
    },
}


payload_json = json.dumps(
    payload,
    separators=(
        ",",
        ":",
    ),
)


# ==========================================================
# 10. HTML
# ==========================================================

html = r"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
F1 Pantheon — __START_YEAR__–__END_YEAR__
</title>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>


<style>

:root {
    --bg: #0d1117;
    --panel: #161b22;
    --panel2: #21262d;
    --border: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --blue: #58a6ff;
    --green: #3fb950;
    --orange: #d29922;
    --red: #f85149;
    --purple: #bc8cff;
}

* {
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

body {
    margin: 0;

    background:
        var(--bg);

    color:
        var(--text);

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    line-height:
        1.45;
}


.page {
    max-width:
        1500px;

    margin:
        0 auto;

    padding:
        28px;
}


h1 {
    margin:
        0;

    font-size:
        clamp(
            30px,
            5vw,
            52px
        );

    letter-spacing:
        -0.045em;
}


h2 {
    margin:
        0 0 16px 0;

    font-size:
        21px;
}


.subtitle {
    color:
        var(--muted);

    margin:
        5px 0 28px 0;
}


.summary-grid {

    display:
        grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                170px,
                1fr
            )
        );

    gap:
        12px;
}


.driver-grid {

    display:
        grid;

    grid-template-columns:
        repeat(
            4,
            minmax(
                0,
                1fr
            )
        );

    gap:
        12px;
}


.sticky-nav {

    position:
        sticky;

    top:
        0;

    z-index:
        50;

    display:
        flex;

    flex-wrap:
        wrap;

    gap:
        8px;

    margin:
        16px 0 2px 0;

    padding:
        8px;

    background:
        rgba(
            13,
            17,
            23,
            0.92
        );

    border:
        1px solid
        var(--border);

    border-radius:
        10px;

    backdrop-filter:
        blur(
            8px
        );
}


.sticky-nav a {

    padding:
        7px 10px;

    color:
        var(--muted);

    text-decoration:
        none;

    border-radius:
        7px;

    font-size:
        12px;

    font-weight:
        650;
}


.sticky-nav a:hover {

    color:
        var(--text);

    background:
        var(--panel2);
}


.check-control {

    display:
        flex;

    flex-direction:
        row;

    align-items:
        center;

    gap:
        7px;

    min-height:
        38px;

    padding:
        0 4px;

    color:
        var(--muted);

    font-size:
        12px;
}


.check-control input {

    margin:
        0;
}


.rank-range-control {

    display:
        flex;

    justify-content:
        flex-end;

    margin:
        -4px 0 6px 0;
}


.metric,
.small-card {

    background:
        var(--panel2);

    border:
        1px solid
        var(--border);

    border-radius:
        10px;

    padding:
        15px;
}


.metric-label,
.small-label {

    color:
        var(--muted);

    font-size:
        11px;

    text-transform:
        uppercase;

    letter-spacing:
        0.08em;
}


.metric-value {

    margin-top:
        5px;

    font-size:
        24px;

    font-weight:
        750;
}


.small-value {

    margin-top:
        3px;

    font-size:
        18px;

    font-weight:
        700;
}


.section {

    margin-top:
        20px;

    padding:
        20px;

    background:
        var(--panel);

    border:
        1px solid
        var(--border);

    border-radius:
        12px;
}


.two-column {

    display:
        grid;

    grid-template-columns:
        minmax(
            0,
            1.15fr
        )
        minmax(
            320px,
            0.85fr
        );

    gap:
        20px;
}


.controls {

    display:
        flex;

    flex-wrap:
        wrap;

    gap:
        10px;

    align-items:
        end;

    margin-bottom:
        15px;
}


.control {

    display:
        flex;

    flex-direction:
        column;

    gap:
        5px;

    min-width:
        175px;
}


label {

    color:
        var(--muted);

    font-size:
        12px;
}


input,
select {

    padding:
        9px 11px;

    color:
        var(--text);

    background:
        var(--panel2);

    border:
        1px solid
        var(--border);

    border-radius:
        7px;

    font-size:
        14px;
}


.table-wrap {

    overflow:
        auto;

    max-height:
        740px;

    border:
        1px solid
        var(--border);

    border-radius:
        8px;
}


table {

    width:
        100%;

    border-collapse:
        collapse;

    font-size:
        13px;
}


th,
td {

    padding:
        9px;

    border-bottom:
        1px solid
        var(--border);

    text-align:
        right;

    white-space:
        nowrap;
}


th {

    position:
        sticky;

    top:
        0;

    z-index:
        2;

    color:
        var(--muted);

    background:
        var(--panel);
}


th:nth-child(2),
td:nth-child(2) {

    text-align:
        left;
}


tbody tr {

    cursor:
        pointer;
}


tbody tr:hover {

    background:
        var(--panel2);
}


.badge {

    display:
        inline-block;

    padding:
        3px 7px;

    border-radius:
        999px;

    background:
        rgba(
            88,
            166,
            255,
            0.14
        );

    color:
        var(--blue);

    font-weight:
        750;
}


.rank-up {
    color:
        var(--green);
}


.rank-down {
    color:
        var(--red);
}


.rank-same {
    color:
        var(--muted);
}


.driver-name {
    font-weight:
        650;
}


.note {

    color:
        var(--muted);

    font-size:
        13px;
}


.chart {

    width:
        100%;

    min-height:
        410px;
}


.compare-result {

    margin-top:
        10px;

    padding:
        24px;

    text-align:
        center;

    background:
        var(--panel2);

    border-radius:
        10px;
}


.probability {

    font-size:
        clamp(
            32px,
            6vw,
            58px
        );

    font-weight:
        800;

    letter-spacing:
        -0.045em;
}


.secondary-probability {

    margin-top:
        6px;

    color:
        var(--muted);
}


.warning {

    margin-top:
        12px;

    padding:
        12px;

    border-left:
        3px solid
        var(--orange);

    background:
        rgba(
            210,
            153,
            34,
            0.08
        );

    color:
        #d8c48c;
}





.season-hint {
    margin: 8px 0 14px 0;
    padding: 10px 12px;
    background: rgba(88,166,255,0.08);
    border-left: 3px solid var(--blue);
    border-radius: 4px;
    color: var(--muted);
    font-size: 13px;
}

.season-detail {
    margin: 8px 0 22px 0;
    padding: 18px;
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 10px;
}

.season-detail-header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 14px;
}

.season-detail-title {
    font-size: 20px;
    font-weight: 750;
}

.season-nav {
    display: flex;
    gap: 8px;
}

button {
    padding: 8px 11px;
    color: var(--text);
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 7px;
    cursor: pointer;
}

button:hover {
    border-color: var(--blue);
}

button:disabled {
    opacity: 0.4;
    cursor: default;
}

.detail-grid {
    display: grid;
    grid-template-columns:
        repeat(
            auto-fit,
            minmax(165px, 1fr)
        );
    gap: 10px;
    margin-bottom: 16px;
}

.detail-table-wrap {
    overflow-x: auto;
    margin-top: 10px;
    border: 1px solid var(--border);
    border-radius: 8px;
}

.detail-heading {
    margin: 18px 0 8px 0;
    font-size: 15px;
    font-weight: 700;
}

.detail-muted {
    color: var(--muted);
}

.evidence-positive {
    color: var(--green);
}

.evidence-negative {
    color: var(--red);
}


footer {

    margin:
        25px 0 8px 0;

    color:
        var(--muted);

    font-size:
        12px;
}




.network-grid {
    display:
        grid;

    grid-template-columns:
        minmax(
            0,
            1.6fr
        )
        minmax(
            320px,
            0.7fr
        );

    gap:
        18px;

    align-items:
        start;
}


.network-chart {
    width:
        100%;

    min-height:
        720px;

    border:
        1px solid
        var(--border);

    border-radius:
        10px;

    overflow:
        hidden;
}


.network-side {
    min-width:
        0;
}


.network-detail {
    background:
        var(--panel2);

    border:
        1px solid
        var(--border);

    border-radius:
        10px;

    padding:
        15px;
}


.network-detail h3 {
    margin:
        0 0 5px 0;

    font-size:
        18px;
}


.network-stat-grid {
    display:
        grid;

    grid-template-columns:
        repeat(
            2,
            minmax(
                0,
                1fr
            )
        );

    gap:
        8px;

    margin:
        14px 0;
}


.network-stat {
    padding:
        10px;

    border:
        1px solid
        var(--border);

    border-radius:
        8px;

    background:
        var(--panel);
}


.network-stat-label {
    color:
        var(--muted);

    font-size:
        10px;

    text-transform:
        uppercase;

    letter-spacing:
        0.07em;
}


.network-stat-value {
    margin-top:
        3px;

    font-size:
        16px;

    font-weight:
        700;
}


.network-table-wrap {
    max-height:
        430px;

    overflow:
        auto;

    border:
        1px solid
        var(--border);

    border-radius:
        8px;
}


.network-table-wrap table {
    font-size:
        12px;
}


.network-table-wrap tbody tr {
    cursor:
        pointer;
}


.network-action-row {
    display:
        flex;

    flex-wrap:
        wrap;

    gap:
        8px;

    margin-top:
        12px;
}


.network-button {
    min-height:
        40px;

    padding:
        8px 12px;

    border:
        1px solid
        var(--border);

    border-radius:
        7px;

    color:
        var(--text);

    background:
        var(--panel2);

    cursor:
        pointer;
}


.network-button:hover {
    border-color:
        var(--blue);
}


.network-status {
    margin-left:
        auto;

    color:
        var(--muted);

    font-size:
        12px;
}


.network-path-panel {
    margin:
        4px 0 16px 0;

    padding:
        12px;

    border:
        1px solid
        var(--border);

    border-radius:
        9px;

    background:
        rgba(88,166,255,0.04);
}


.network-path-panel .controls {
    margin-bottom:
        8px;
}


.network-path-status {
    color:
        var(--muted);

    font-size:
        12px;

    line-height:
        1.55;
}


.network-path-route {
    display:
        flex;

    flex-wrap:
        wrap;

    align-items:
        center;

    gap:
        6px;

    margin-top:
        8px;
}


.network-path-node {
    padding:
        5px 8px;

    border:
        1px solid
        rgba(210,153,34,0.55);

    border-radius:
        999px;

    background:
        rgba(210,153,34,0.10);

    color:
        var(--text);

    cursor:
        pointer;

    font-size:
        12px;
}


.network-path-node:hover {
    border-color:
        var(--orange);
}


.network-path-edge-label {
    color:
        var(--orange);

    font-size:
        11px;

    white-space:
        nowrap;
}



/* ======================================================
   TEAMMATE CHAIN GAME
====================================================== */

.game-shell {
    display:
        grid;

    grid-template-columns:
        minmax(
            0,
            1.35fr
        )
        minmax(
            280px,
            0.65fr
        );

    gap:
        18px;

    align-items:
        stretch;
}


.game-board,
.game-info-card {
    border:
        1px solid
        var(--border);

    border-radius:
        14px;

    background:
        rgba(
            13,
            17,
            23,
            0.55
        );

    padding:
        18px;
}


.game-endpoints {
    display:
        grid;

    grid-template-columns:
        minmax(
            0,
            1fr
        )
        auto
        minmax(
            0,
            1fr
        );

    gap:
        12px;

    align-items:
        center;

    margin:
        16px 0;
}


.game-endpoint {
    border:
        1px solid
        var(--border);

    border-radius:
        12px;

    padding:
        14px;

    background:
        var(--panel);
}


.game-endpoint-label {
    color:
        var(--muted);

    text-transform:
        uppercase;

    letter-spacing:
        0.08em;

    font-size:
        10px;

    margin-bottom:
        5px;
}


.game-endpoint-name {
    color:
        var(--text);

    font-size:
        18px;

    font-weight:
        700;
}


.game-arrow {
    color:
        var(--orange);

    font-size:
        26px;

    font-weight:
        700;
}


.game-difficulty-row {
    display:
        flex;

    flex-wrap:
        wrap;

    gap:
        8px;

    align-items:
        center;

    margin-bottom:
        14px;
}


.game-badge {
    display:
        inline-flex;

    align-items:
        center;

    gap:
        5px;

    padding:
        5px
        9px;

    border-radius:
        999px;

    border:
        1px solid
        var(--border);

    background:
        rgba(
            22,
            27,
            34,
            0.85
        );

    font-size:
        12px;
}


.game-badge[data-tier="Easy"] {
    border-color:
        rgba(
            63,
            185,
            80,
            0.65
        );
}


.game-badge[data-tier="Medium"] {
    border-color:
        rgba(
            88,
            166,
            255,
            0.65
        );
}


.game-badge[data-tier="Hard"] {
    border-color:
        rgba(
            210,
            153,
            34,
            0.75
        );
}


.game-badge[data-tier="Expert"] {
    border-color:
        rgba(
            248,
            81,
            73,
            0.75
        );
}


.game-badge[data-tier="Nightmare"] {
    border-color:
        rgba(
            188,
            140,
            255,
            0.80
        );
}


.game-route {
    min-height:
        68px;

    display:
        flex;

    flex-wrap:
        wrap;

    align-items:
        center;

    gap:
        7px;

    padding:
        12px;

    border:
        1px solid
        var(--border);

    border-radius:
        12px;

    background:
        rgba(
            1,
            4,
            9,
            0.35
        );

    margin:
        12px 0;
}


.game-route-node {
    display:
        inline-flex;

    align-items:
        center;

    padding:
        6px
        10px;

    border-radius:
        999px;

    border:
        1px solid
        rgba(
            88,
            166,
            255,
            0.45
        );

    background:
        rgba(
            88,
            166,
            255,
            0.10
        );

    font-size:
        12px;

    font-weight:
        600;
}


.game-route-edge {
    color:
        var(--muted);

    font-size:
        11px;
}


.game-input-row {
    display:
        grid;

    grid-template-columns:
        minmax(
            0,
            1fr
        )
        auto;

    gap:
        8px;

    align-items:
        end;
}


.game-input-row input {
    width:
        100%;
}


.game-actions {
    display:
        flex;

    flex-wrap:
        wrap;

    gap:
        8px;

    margin-top:
        10px;
}


.game-status {
    margin-top:
        12px;

    min-height:
        42px;

    padding:
        10px
        12px;

    border-left:
        3px solid
        var(--border);

    background:
        rgba(
            22,
            27,
            34,
            0.60
        );

    border-radius:
        0
        8px
        8px
        0;

    font-size:
        13px;
}


.game-status.success {
    border-left-color:
        var(--green);
}


.game-status.error {
    border-left-color:
        var(--red);
}


.game-status.hint {
    border-left-color:
        var(--orange);
}


.game-stats {
    display:
        grid;

    grid-template-columns:
        repeat(
            3,
            minmax(
                0,
                1fr
            )
        );

    gap:
        8px;

    margin-top:
        12px;
}


.game-stat {
    border:
        1px solid
        var(--border);

    border-radius:
        10px;

    padding:
        10px;

    text-align:
        center;
}


.game-stat-value {
    font-size:
        18px;

    font-weight:
        700;

    color:
        var(--text);
}


.game-stat-label {
    margin-top:
        3px;

    color:
        var(--muted);

    font-size:
        10px;

    text-transform:
        uppercase;

    letter-spacing:
        0.06em;
}


.game-complete {
    display:
        none;

    margin-top:
        14px;

    border:
        1px solid
        rgba(
            63,
            185,
            80,
            0.55
        );

    border-radius:
        12px;

    padding:
        14px;

    background:
        rgba(
            63,
            185,
            80,
            0.07
        );
}


.game-complete.visible {
    display:
        block;
}


.game-score {
    font-size:
        30px;

    font-weight:
        800;

    color:
        var(--text);

    margin-bottom:
        4px;
}


.game-optimal-route {
    margin-top:
        10px;

    color:
        var(--muted);

    line-height:
        1.7;
}


.game-info-card h3 {
    margin-top:
        0;
}


.game-info-list {
    margin:
        0;

    padding-left:
        18px;

    color:
        var(--muted);

    line-height:
        1.7;
}


.game-local-best {
    margin-top:
        14px;

    padding-top:
        12px;

    border-top:
        1px solid
        var(--border);

    font-size:
        12px;

    color:
        var(--muted);
}


.game-small {
    color:
        var(--muted);

    font-size:
        11px;

    line-height:
        1.6;
}


@media (
    max-width:
        1150px
) {

    .driver-grid {

        grid-template-columns:
            repeat(
                2,
                minmax(
                    0,
                    1fr
                )
            );
    }

}


@media (
    max-width:
        900px
) {

    .page {

        padding:
            14px;
    }


    .two-column {

        grid-template-columns:
            1fr;
    }


    .network-grid {

        grid-template-columns:
            1fr;
    }


    .game-shell {

        grid-template-columns:
            1fr;
    }


    .network-chart {

        min-height:
            560px;
    }

}


@media (
    max-width:
        560px
) {

    .driver-grid {

        grid-template-columns:
            1fr;
    }


    .game-endpoints {

        grid-template-columns:
            1fr;
    }


    .game-arrow {

        transform:
            rotate(
                90deg
            );

        text-align:
            center;
    }


    .game-input-row {

        grid-template-columns:
            1fr;
    }

}

</style>

</head>


<body>

<div class="page">


<h1>
F1 Pantheon
</h1>


<div class="subtitle">

Bayesian driver–constructor ranking,
__START_YEAR__–__END_YEAR__

&nbsp;•&nbsp;

60% five-season peak

&nbsp;•&nbsp;

40% square-root-damped longevity

</div>


<nav class="sticky-nav" aria-label="Dashboard sections">

<a href="#leaderboard-section">Leaderboard</a>
<a href="#driver-explorer-section">Driver explorer</a>
<a href="#network-section">Teammate network</a>
<a href="#game-section">Teammate Chain</a>
<a href="#compare-section">Compare</a>
<a href="#method-section">Method</a>

</nav>


<div
    id="summary"
    class="summary-grid">
</div>


<!-- ======================================================
     LEADERBOARD
======================================================= -->

<div id="leaderboard-section" class="section">

<h2>
Pantheon leaderboard
</h2>


<div class="controls">


<div class="control">

<label for="search">
Search driver
</label>

<input
    id="search"
    type="text"
    placeholder="Hamilton, Senna..."
>

</div>


<div class="control">

<label for="sort">
Sort
</label>


<select id="sort">

<option value="rank">
Posterior mean JAWS
</option>

<option value="rank_mean">
Expected posterior rank
</option>

<option value="rank_lb">
Rank lower bound
</option>

<option value="rank_ub">
Rank upper bound
</option>

<option value="rank_width">
Rank interval width
</option>

<option value="peak_score">
Peak score
</option>

<option value="longevity_score">
Longevity score
</option>

<option value="p_rank_1">
P(rank 1)
</option>

<option value="starts">
Career starts
</option>

</select>

</div>


<div class="control">

<label for="topn">
Show
</label>


<select id="topn">

<option value="20">
Top 20
</option>

<option
    value="30"
    selected>
Top 30
</option>

<option value="50">
Top 50
</option>

<option value="999">
All
</option>

</select>

</div>


</div>


<div class="table-wrap">

<table>

<thead>

<tr>

<th>
Rank
</th>

<th>
Driver
</th>

<th>
JAWS
</th>

<th>
89% JAWS interval
</th>

<th>
Expected rank
</th>

<th>
Peak
</th>

<th>
Longevity
</th>

<th>
Rank LB
</th>

<th>
Rank UB
</th>

<th>
Starts
</th>

</tr>

</thead>


<tbody id="leaderboard-body">
</tbody>

</table>

</div>

</div>


<!-- ======================================================
     DRIVER EXPLORER
======================================================= -->

<div id="driver-explorer-section" class="section">

<h2>
Driver explorer
</h2>


<div class="controls">

<div class="control">

<label for="driver-select">
Driver
</label>

<select id="driver-select">
</select>

</div>

</div>


<div
    id="driver-cards"
    class="driver-grid">
</div>


<div
    id="trajectory"
    class="chart">
</div>


<div class="season-hint">
Click any season point in the trajectory to inspect that
year's driver ability, constructor/car performance and
teammate comparison. Use the previous/next buttons to move
through the career.
</div>


<div
    id="season-detail"
    class="season-detail">
</div>


<div class="rank-range-control">

<label class="check-control" for="rank-full-range">

<input
    id="rank-full-range"
    type="checkbox">

Show full rank range

</label>

</div>


<div
    id="rank-dist"
    class="chart">
</div>

</div>


<!-- ======================================================
     ENTIRE TEAMMATE NETWORK
======================================================= -->

<div id="network-section" class="section">

<h2>
Interactive 3D teammate network
</h2>


<p class="note">

Every node is a modelled F1 driver from 1982–2025 and every
edge represents shared actual race starts for the same
constructor. Node colour is mean modelled-season α, node
size reflects actual starts, and stronger edges represent
more shared starts. The horizontal coordinates are
force-directed network-layout coordinates; the vertical
axis is the driver's modelled-career midpoint year, so the
third dimension separates eras.

</p>


<div class="controls">

<div class="control">

<label for="network-driver-select">
Focus driver
</label>

<select id="network-driver-select">
</select>

</div>


<div class="control">

<label for="network-min-starts">
Minimum shared starts per edge
</label>

<select id="network-min-starts">

<option value="1">
1 — show every edge
</option>

<option value="3">
3+
</option>

<option value="5" selected>
5+
</option>

<option value="10">
10+
</option>

<option value="20">
20+
</option>

<option value="40">
40+
</option>

</select>

</div>


<div class="control">

<label for="network-view-mode">
Network view
</label>

<select id="network-view-mode">

<option value="largest" selected>
Largest connected component
</option>

<option value="full">
Entire network
</option>

<option value="ego">
Selected driver's neighbourhood
</option>

<option value="path">
Shortest path
</option>

</select>

</div>


<label class="check-control" for="network-show-all-teammates">

<input
    id="network-show-all-teammates"
    type="checkbox">

Show all teammates in table

</label>


<button
    id="network-reset-camera"
    class="network-button"
    type="button">
Reset 3D view
</button>


<div
    id="network-status"
    class="network-status">
</div>

</div>


<div class="network-path-panel">

<div class="controls">

<div class="control">

<label for="network-path-a">
Shortest path — Driver A
</label>

<select id="network-path-a">
</select>

</div>


<div class="control">

<label for="network-path-b">
Shortest path — Driver B
</label>

<select id="network-path-b">
</select>

</div>


<button
    id="network-find-path"
    class="network-button"
    type="button">
Find shortest path
</button>


<button
    id="network-clear-path"
    class="network-button"
    type="button">
Clear path
</button>

</div>


<div
    id="network-path-status"
    class="network-path-status">
Choose two drivers to find the minimum number of teammate
edges connecting them. The path uses the current minimum
shared-start threshold; ties favour stronger shared-start
connections.
</div>

</div>


<div class="network-grid">

<div
    id="teammate-network"
    class="network-chart">
</div>


<div class="network-side">

<div
    id="network-detail"
    class="network-detail">
</div>

</div>

</div>


<p class="note" style="margin-top:12px;">

Drag to rotate, scroll to zoom and click any node to inspect
its direct teammate connections. In neighbourhood mode,
driver names are shown directly on the graph. The default
largest-component view avoids detached components compressing
the main network. The shortest-path tool finds the minimum
number of teammate links between any two drivers at the
current shared-start threshold and highlights that route in
the 3D graph. Pantheon-eligible nodes can be opened directly
in the driver explorer.

</p>

</div>



<!-- ======================================================
     TEAMMATE CHAIN GAME
======================================================= -->

<div id="game-section" class="section">

<h2>
Teammate Chain
</h2>


<p class="note">

Connect the two drivers using only real F1 teammate links.
Every game edge means the two drivers made at least one actual
race start for the same constructor. Fewer links are better.
The challenge difficulty combines minimum chain length,
endpoint recognition, how historical the drivers are and how
many equally-short routes exist.

</p>


<div class="controls">

<div class="control">

<label for="game-mode">
Challenge mode
</label>

<select id="game-mode">

<option value="daily" selected>
Daily challenge
</option>

<option value="random">
Random challenge
</option>

</select>

</div>


<div class="control">

<label for="game-difficulty">
Random difficulty
</label>

<select id="game-difficulty">

<option value="Any">
Any
</option>

<option value="Easy">
Easy
</option>

<option value="Medium">
Medium
</option>

<option value="Hard" selected>
Hard
</option>

<option value="Expert">
Expert
</option>

<option value="Nightmare">
Nightmare
</option>

</select>

</div>


<button
    id="game-new"
    class="network-button"
    type="button">
New challenge
</button>

</div>


<div class="game-shell">

<div class="game-board">

<div class="game-endpoints">

<div class="game-endpoint">

<div class="game-endpoint-label">
Start
</div>

<div
    id="game-start-name"
    class="game-endpoint-name">
</div>

</div>


<div class="game-arrow">
→
</div>


<div class="game-endpoint">

<div class="game-endpoint-label">
Target
</div>

<div
    id="game-target-name"
    class="game-endpoint-name">
</div>

</div>

</div>


<div class="game-difficulty-row">

<span
    id="game-difficulty-badge"
    class="game-badge">
</span>

<span
    id="game-daily-label"
    class="game-badge">
</span>

</div>


<div
    id="game-route"
    class="game-route">
</div>


<div class="game-input-row">

<div class="control" style="margin:0;">

<label for="game-next-driver">
Next teammate
</label>

<input
    id="game-next-driver"
    type="text"
    list="game-driver-options"
    autocomplete="off"
    placeholder="Start typing a driver name...">

<datalist id="game-driver-options">
</datalist>

</div>


<button
    id="game-submit"
    class="network-button"
    type="button">
Add driver
</button>

</div>


<div class="game-actions">

<button
    id="game-undo"
    class="network-button"
    type="button">
Undo
</button>

<button
    id="game-restart"
    class="network-button"
    type="button">
Restart
</button>

<button
    id="game-hint"
    class="network-button"
    type="button">
Hint
</button>

<button
    id="game-give-up"
    class="network-button"
    type="button">
Give up
</button>

</div>


<div
    id="game-status"
    class="game-status">
</div>


<div class="game-stats">

<div class="game-stat">

<div
    id="game-links-used"
    class="game-stat-value">
0
</div>

<div class="game-stat-label">
Links used
</div>

</div>


<div class="game-stat">

<div
    id="game-invalid"
    class="game-stat-value">
0
</div>

<div class="game-stat-label">
Invalid guesses
</div>

</div>


<div class="game-stat">

<div
    id="game-hints"
    class="game-stat-value">
0
</div>

<div class="game-stat-label">
Hints
</div>

</div>

</div>


<div
    id="game-complete"
    class="game-complete">

<div
    id="game-score"
    class="game-score">
</div>

<div
    id="game-result-summary">
</div>

<div
    id="game-optimal-route"
    class="game-optimal-route">
</div>

<div class="game-actions">

<button
    id="game-view-route"
    class="network-button"
    type="button">
View my route in 3D
</button>

<button
    id="game-view-optimal"
    class="network-button"
    type="button">
View optimal route in 3D
</button>

</div>

</div>

</div>


<div class="game-info-card">

<h3>
How difficulty works
</h3>

<ul class="game-info-list">

<li>
50% — minimum teammate-chain length.
</li>

<li>
25% — endpoint obscurity, using starts, wins and podiums as a
recognition proxy rather than model ability.
</li>

<li>
15% — how historical the two drivers are.
</li>

<li>
10% — route scarcity: one unique shortest path is harder than
having several equally-short solutions.
</li>

</ul>


<p class="game-small">

Generated challenges never use direct teammates as endpoints,
so every puzzle requires at least one intermediate driver.
The game always uses the complete 1+ shared-start teammate
graph, regardless of the edge threshold selected in the 3D
network above.

</p>


<div
    id="game-local-best"
    class="game-local-best">
</div>

</div>

</div>

</div>


<!-- ======================================================
     PEAK / LONGEVITY + HEAD TO HEAD
======================================================= -->

<div id="compare-section" class="two-column">


<div class="section">

<h2>
Peak vs longevity
</h2>


<div
    id="peak-longevity"
    class="chart">
</div>


<p class="note">

Each point is a Pantheon-eligible driver.
Click a point to open that driver in the explorer.

</p>

</div>


<div class="section">

<h2>
Head-to-head posterior
</h2>


<div class="controls">


<div class="control">

<label for="compare-a">
Driver A
</label>

<select id="compare-a">
</select>

</div>


<div class="control">

<label for="compare-b">
Driver B
</label>

<select id="compare-b">
</select>

</div>


</div>


<div
    id="compare-result"
    class="compare-result">
</div>


<p class="note">

The probability is

<strong>
P(JAWS A &gt; JAWS B)
</strong>

across the 4,000 posterior draws.

Values near 50% indicate substantial overlap rather than a
meaningful ordering between the two drivers.

</p>

</div>


</div>


<!-- ======================================================
     MODEL NOTES
======================================================= -->

<div id="method-section" class="section">

<h2>
How to read the model
</h2>


<p>

The season-level latent value

<strong>α</strong>

represents realised driver ability relative to the average
occupied F1 seat in that season, after separating
constructor performance and constructor-tenure context.

Age-related decline remains part of realised driver ability.

</p>


<p>

The final Pantheon score combines

<strong>60% peak</strong>

— the mean of the driver's best five latent seasons —

with

<strong>40% longevity</strong>

— the square root of cumulative positive career α surplus.

Peak and longevity are independently normalised to 100
within every posterior draw before combination.

</p>


<p>

Eligibility requires at least

<strong>35 actual race starts</strong>,

<strong>3 unique actual-start teammates</strong>,

and

<strong>5 modelled seasons</strong>.

</p>


<div class="warning">

<strong>
Historical coverage:
</strong>

the model begins in 1982.

Career-surplus and potentially peak values are therefore
truncated for drivers whose Formula One careers began before
1982.

In particular, this affects the interpretation of drivers
such as Prost, Mansell, Piquet, Keke Rosberg, Patrese and
de Angelis.

The ranking should therefore be described as the

<strong>
1982–2025 F1 Pantheon
</strong>,

rather than a complete-career ranking of all Formula One
history.

</div>


<p class="note">

The final production posterior contains 4,000 draws.

The production fit passed its convergence diagnostics with
zero divergences and no R-hat values above 1.01.

Rank LB and Rank UB are the 5.5th and 94.5th percentiles of
the posterior rank distribution, forming an 89% posterior
rank interval.

Clicking a season in the driver explorer opens the
season-level context. Driver α and constructor β are both
centred within that season. The teammate Δα is
selected-driver α minus teammate α. For modern seasons the
pace and qualifying contrast columns are descriptive
summaries of the standardised teammate-contrast observations
used by the likelihood; positive values favour the selected
driver.

The constructor β term should be read as constructor-season
performance context (car plus constructor/team effects), not
as a mechanically isolated chassis measurement.

</p>

</div>


<footer>

Final validated joint driver–constructor model.

Posterior uncertainty is propagated through the final
driver-season states, JAWS scores and ranks.

</footer>


</div>


<script>

const DATA =
    __PANTHEON_DATA__;


const tableData =
    DATA.leaderboard;


const byDriver =
    Object.fromEntries(

        tableData.map(

            row => [
                row.driver,
                row
            ]

        )

    );


const driverIndex =
    Object.fromEntries(

        DATA.drivers.map(

            (
                driver,
                i
            ) => [
                driver,
                i
            ]

        )

    );


const networkNodes =
    DATA.network.nodes;


const networkEdges =
    DATA.network.edges;


const teammateGame =
    DATA.game;


const gameDriverMetrics =
    teammateGame.drivers;


const gameChallenges =
    teammateGame.challenges;


const networkNodeByDriver =
    Object.fromEntries(

        networkNodes.map(

            node => [
                node.driver,
                node
            ]

        )

    );


let currentNetworkDriver = null;

let networkReady = false;

let currentNetworkPath = null;

let currentNetworkPathA = null;

let currentNetworkPathB = null;


let currentGameChallenge = null;

let currentGameRoute = [];

let gameInvalidGuesses = 0;

let gameHintsUsed = 0;

let gameSolved = false;

let gameGaveUp = false;

let gameLastMessage = null;


const plotConfig = {

    responsive:
        true,

    displaylogo:
        false

};


function layoutBase() {

    return {

        paper_bgcolor:
            "#161b22",

        plot_bgcolor:
            "#161b22",

        font: {

            color:
                "#c9d1d9"

        },

        margin: {

            l:
                58,

            r:
                25,

            t:
                45,

            b:
                55

        },

        xaxis: {

            gridcolor:
                "#30363d",

            zerolinecolor:
                "#484f58"

        },

        yaxis: {

            gridcolor:
                "#30363d",

            zerolinecolor:
                "#484f58"

        },

        hoverlabel: {

            bgcolor:
                "#21262d",

            bordercolor:
                "#30363d",

            font: {

                color:
                    "#e6edf3"

            }

        }

    };

}


function pct(
    value
) {

    return (
        100 * value
    ).toFixed(
        1
    ) + "%";

}


function num(
    value,
    digits=1
) {

    return Number(
        value
    ).toFixed(
        digits
    );

}


function initialiseSummary() {

    const top =
        tableData[
            0
        ];

    document.getElementById(
        "summary"
    ).innerHTML = `

        <div class="metric">

            <div class="metric-label">
                Pantheon drivers
            </div>

            <div class="metric-value">
                ${DATA.metadata.n_drivers}
            </div>

        </div>


        <div class="metric">

            <div class="metric-label">
                Posterior draws
            </div>

            <div class="metric-value">
                ${DATA.metadata.n_draws.toLocaleString()}
            </div>

        </div>


        <div class="metric">

            <div class="metric-label">
                Posterior mean #1
            </div>

            <div class="metric-value">
                ${top.display_name}
            </div>

        </div>


        <div class="metric">

            <div class="metric-label">
                Coverage
            </div>

            <div class="metric-value">
                ${DATA.metadata.start_year}–${DATA.metadata.end_year}
            </div>

        </div>


        <div class="metric">

            <div class="metric-label">
                Eligibility
            </div>

            <div class="metric-value">
                35 / 3 / 5
            </div>

            <div class="note">
                starts / teammates / seasons
            </div>

        </div>

    `;

}


function populateSelect(
    element,
    selectedDriver=null
) {

    element.innerHTML =
        "";


    tableData.forEach(

        row => {

            const option =
                document.createElement(
                    "option"
                );


            option.value =
                row.driver;


            option.textContent =
                `${row.rank}. ${row.display_name}`;


            if (
                row.driver
                === selectedDriver
            ) {

                option.selected =
                    true;

            }


            element.appendChild(
                option
            );

        }

    );

}


function renderLeaderboard() {

    const search =
        document.getElementById(
            "search"
        )
        .value
        .trim()
        .toLowerCase();


    const sortKey =
        document.getElementById(
            "sort"
        ).value;


    const limit =
        Number(
            document.getElementById(
                "topn"
            ).value
        );


    let rows = [
        ...tableData
    ];


    if (
        search
    ) {

        rows =
            rows.filter(

                row =>
                    row.display_name
                    .toLowerCase()
                    .includes(
                        search
                    )

            );

    }


    const ascending =
        new Set([

            "rank",

            "rank_mean",

            "rank_lb",

            "rank_ub"

        ]);


    rows.sort(

        (
            a,
            b
        ) => {

            if (
                ascending.has(
                    sortKey
                )
            ) {

                return (
                    a[
                        sortKey
                    ]
                    -
                    b[
                        sortKey
                    ]
                );

            }


            return (
                b[
                    sortKey
                ]
                -
                a[
                    sortKey
                ]
            );

        }

    );


    rows =
        rows.slice(
            0,
            limit
        );


    const tbody =
        document.getElementById(
            "leaderboard-body"
        );


    tbody.innerHTML =
        rows.map(

            row => `

            <tr data-driver="${row.driver}">

                <td>
                    #${row.rank}
                </td>

                <td class="driver-name">
                    ${row.display_name}
                </td>

                <td>

                    <span class="badge">
                        ${num(row.jaws_mean)}
                    </span>

                </td>

                <td>

                    ${num(row.jaws_lb)}
                    –
                    ${num(row.jaws_ub)}

                </td>

                <td>
                    #${num(row.rank_mean, 1)}
                </td>

                <td>
                    ${num(row.peak_score)}
                </td>

                <td>
                    ${num(row.longevity_score)}
                </td>

                <td>
                    #${num(row.rank_lb, 0)}
                </td>

                <td>
                    #${num(row.rank_ub, 0)}
                </td>

                <td>
                    ${row.starts}
                </td>

            </tr>

        `).join(
            ""
        );


    tbody.querySelectorAll(
        "tr"
    ).forEach(

        element => {

            element.addEventListener(

                "click",

                () => {

                    const driver =
                        element.dataset.driver;


                    document.getElementById(
                        "driver-select"
                    ).value =
                        driver;


                    renderDriver(
                        driver
                    );


                    document.getElementById(
                        "driver-select"
                    ).scrollIntoView({

                        behavior:
                            "smooth",

                        block:
                            "center"

                    });

                }

            );

        }

    );

}


let currentDriver = null;

let currentSeason = null;


function signedNum(
    value,
    digits=3
) {

    if (
        value === null
        ||
        value === undefined
        ||
        Number.isNaN(
            Number(
                value
            )
        )
    ) {

        return "—";
    }

    const x =
        Number(
            value
        );

    return (
        x >= 0
        ? "+"
        : ""
    ) + x.toFixed(
        digits
    );
}


function intervalText(
    lb,
    ub,
    digits=3
) {

    if (
        lb === null
        ||
        ub === null
        ||
        lb === undefined
        ||
        ub === undefined
    ) {

        return "—";
    }

    return (
        `${Number(lb).toFixed(digits)}`
        + " – "
        + `${Number(ub).toFixed(digits)}`
    );
}


function evidenceText(
    evidence
) {

    if (
        !evidence
    ) {

        return "—";
    }

    return (
        `${signedNum(
            evidence.mean,
            3
        )}`
        + ` <span class="note">`
        + `(${evidence.n_weekends} wkds)`
        + `</span>`
    );
}


function updateUrlSelection(
    driver,
    year
) {

    const params =
        new URLSearchParams();

    params.set(
        "driver",
        driver
    );

    params.set(
        "year",
        year
    );

    history.replaceState(
        null,
        "",
        `#${params.toString()}`
    );
}


function readUrlSelection() {

    const params =
        new URLSearchParams(
            window.location.hash.replace(
                /^#/,
                ""
            )
        );

    const driver =
        params.get(
            "driver"
        );

    const year =
        Number(
            params.get(
                "year"
            )
        );

    return {
        driver:
            driver,
        year:
            Number.isFinite(
                year
            )
            ? year
            : null
    };
}


function renderSeasonDetail(
    driver,
    year
) {

    const detail =
        DATA.seasons[
            driver
        ][
            String(
                year
            )
        ];

    if (
        !detail
    ) {

        document.getElementById(
            "season-detail"
        ).innerHTML = `
            <div class="detail-muted">
                No season detail is available for ${year}.
            </div>
        `;

        return;
    }


    currentDriver =
        driver;

    currentSeason =
        Number(
            year
        );


    updateUrlSelection(
        driver,
        year
    );


    const row =
        byDriver[
            driver
        ];


    const years =
        DATA.trajectories[
            driver
        ].years;


    const yearIndex =
        years.indexOf(
            Number(
                year
            )
        );


    const hasPrev =
        yearIndex > 0;


    const hasNext =
        yearIndex
        <
        years.length - 1;


    const constructorRows =
        detail.teams.map(
            team => {

                const car =
                    team.car;

                let carEffect =
                    "—";

                let carInterval =
                    "—";

                let carRank =
                    "—";


                if (
                    car
                ) {

                    carEffect =
                        signedNum(
                            car.mean,
                            3
                        );

                    carInterval =
                        intervalText(
                            car.lb,
                            car.ub,
                            3
                        );

                    if (
                        car.rank
                    ) {

                        carRank =
                            `#${num(
                                car.rank.mean,
                                1
                            )}`
                            + ` `
                            + `<span class="note">`
                            + `(#${num(
                                car.rank.lb,
                                0
                            )}–#${num(
                                car.rank.ub,
                                0
                            )})`
                            + `</span>`;
                    }
                }


                return `

                    <tr>

                        <td style="text-align:left;">
                            ${team.constructor_name}
                        </td>

                        <td>
                            ${team.starts}
                        </td>

                        <td>
                            ${carEffect}
                        </td>

                        <td>
                            ${carInterval}
                        </td>

                        <td>
                            ${carRank}
                        </td>

                    </tr>

                `;

            }
        ).join(
            ""
        );


    const teammateRows =
        detail.teams.flatMap(
            team =>
                team.teammates.map(
                    teammate => {

                        const deltaClass =
                            (
                                teammate.delta_mean
                                === null
                                ||
                                teammate.delta_mean
                                === undefined
                            )
                            ? ""
                            : (
                                teammate.delta_mean
                                >= 0
                                ? "evidence-positive"
                                : "evidence-negative"
                            );


                        const teammateAlpha =
                            (
                                teammate.alpha_mean
                                === null
                                ||
                                teammate.alpha_mean
                                === undefined
                            )
                            ? "—"
                            : (
                                `${signedNum(
                                    teammate.alpha_mean,
                                    3
                                )}`
                                + ` <span class="note">`
                                + `[${intervalText(
                                    teammate.alpha_lb,
                                    teammate.alpha_ub,
                                    3
                                )}]`
                                + `</span>`
                            );


                        const deltaAlpha =
                            (
                                teammate.delta_mean
                                === null
                                ||
                                teammate.delta_mean
                                === undefined
                            )
                            ? "—"
                            : (
                                `<span class="${deltaClass}">`
                                + `${signedNum(
                                    teammate.delta_mean,
                                    3
                                )}`
                                + `</span>`
                                + ` <span class="note">`
                                + `[${intervalText(
                                    teammate.delta_lb,
                                    teammate.delta_ub,
                                    3
                                )}]`
                                + `</span>`
                            );


                        const probability =
                            (
                                teammate.p_driver_better
                                === null
                                ||
                                teammate.p_driver_better
                                === undefined
                            )
                            ? "—"
                            : pct(
                                teammate.p_driver_better
                            );


                        return `

                            <tr>

                                <td style="text-align:left;">
                                    ${team.constructor_name}
                                </td>

                                <td style="text-align:left;">
                                    ${teammate.display_name}
                                </td>

                                <td>
                                    ${teammate.shared_starts}
                                </td>

                                <td>
                                    ${teammateAlpha}
                                </td>

                                <td>
                                    ${deltaAlpha}
                                </td>

                                <td>
                                    ${probability}
                                </td>

                                <td>
                                    ${evidenceText(
                                        teammate.pace_evidence
                                    )}
                                </td>

                                <td>
                                    ${evidenceText(
                                        teammate.quali_evidence
                                    )}
                                </td>

                            </tr>

                        `;

                    }
                )
        ).join(
            ""
        );


    const noConstructorRows =
        detail.teams.length
        === 0;


    const noTeammateRows =
        detail.teams.every(
            team =>
                team.teammates.length
                === 0
        );


    document.getElementById(
        "season-detail"
    ).innerHTML = `

        <div class="season-detail-header">

            <div>

                <div class="season-detail-title">
                    ${row.display_name} — ${year}
                </div>

                <div class="note">
                    ${detail.era}
                </div>

            </div>


            <div class="season-nav">

                <button
                    id="prev-season"
                    ${hasPrev ? "" : "disabled"}>
                    ← Previous
                </button>

                <button
                    id="next-season"
                    ${hasNext ? "" : "disabled"}>
                    Next →
                </button>

            </div>

        </div>


        <div class="detail-grid">

            <div class="small-card">

                <div class="small-label">
                    Driver α
                </div>

                <div class="small-value">
                    ${signedNum(
                        detail.driver_alpha.mean,
                        3
                    )}
                </div>

                <div class="note">
                    89%:
                    ${intervalText(
                        detail.driver_alpha.lb,
                        detail.driver_alpha.ub,
                        3
                    )}
                </div>

            </div>


            <div class="small-card">

                <div class="small-label">
                    Expected season rank
                </div>

                <div class="small-value">
                    #${num(
                        detail.driver_rank.mean,
                        1
                    )}
                </div>

                <div class="note">
                    89%:
                    #${num(
                        detail.driver_rank.lb,
                        0
                    )}
                    –
                    #${num(
                        detail.driver_rank.ub,
                        0
                    )}
                    of
                    ${detail.driver_rank.field_size}
                </div>

            </div>


            <div class="small-card">

                <div class="small-label">
                    P(α &gt; 0)
                </div>

                <div class="small-value">
                    ${pct(
                        detail.driver_alpha.p_gt_zero
                    )}
                </div>

                <div class="note">
                    relative to average occupied seat
                </div>

            </div>


            <div class="small-card">

                <div class="small-label">
                    Constructor stints
                </div>

                <div class="small-value">
                    ${detail.teams.length}
                </div>

                <div class="note">
                    actual-start team assignments
                </div>

            </div>

        </div>


        <div class="detail-heading">
            Constructor / car context
        </div>


        ${
            noConstructorRows
            ? `
                <div class="detail-muted">
                    No actual-start constructor assignment is available
                    for this modelled season.
                </div>
            `
            : `
                <div class="detail-table-wrap">

                    <table>

                        <thead>

                            <tr>

                                <th style="text-align:left;">
                                    Constructor
                                </th>

                                <th>
                                    Starts
                                </th>

                                <th>
                                    β
                                </th>

                                <th>
                                    89% β interval
                                </th>

                                <th>
                                    Expected constructor rank
                                </th>

                            </tr>

                        </thead>

                        <tbody>
                            ${constructorRows}
                        </tbody>

                    </table>

                </div>
            `
        }


        <div class="detail-heading">
            Teammate comparison
        </div>


        ${
            noTeammateRows
            ? `
                <div class="detail-muted">
                    No shared actual-start teammate is available
                    for this season.
                </div>
            `
            : `
                <div class="detail-table-wrap">

                    <table>

                        <thead>

                            <tr>

                                <th style="text-align:left;">
                                    Constructor
                                </th>

                                <th style="text-align:left;">
                                    Teammate
                                </th>

                                <th>
                                    Shared starts
                                </th>

                                <th>
                                    Teammate α
                                </th>

                                <th>
                                    Δα
                                </th>

                                <th>
                                    P(driver &gt; teammate)
                                </th>

                                <th>
                                    Pace Δ
                                </th>

                                <th>
                                    Quali Δ
                                </th>

                            </tr>

                        </thead>

                        <tbody>
                            ${teammateRows}
                        </tbody>

                    </table>

                </div>
            `
        }


        <p class="note" style="margin-top:12px;">

            Δα is the selected driver's season α minus the
            teammate's season α.

            Constructor β is the modelled constructor-season
            context; higher is stronger.

            For 1996 onward, Pace Δ and Quali Δ are weighted
            summaries of the standardised teammate-contrast
            observations used by the modern likelihood.
            Positive values favour ${row.display_name}.

            Historical seasons use ordinal grid/race evidence,
            so those direct modern contrast columns are blank.

        </p>

    `;


    const prevButton =
        document.getElementById(
            "prev-season"
        );


    const nextButton =
        document.getElementById(
            "next-season"
        );


    if (
        prevButton
    ) {

        prevButton.addEventListener(
            "click",
            () => {

                if (
                    hasPrev
                ) {

                    renderDriver(
                        driver,
                        years[
                            yearIndex - 1
                        ]
                    );
                }
            }
        );
    }


    if (
        nextButton
    ) {

        nextButton.addEventListener(
            "click",
            () => {

                if (
                    hasNext
                ) {

                    renderDriver(
                        driver,
                        years[
                            yearIndex + 1
                        ]
                    );
                }
            }
        );
    }

}


function renderDriver(
    driver,
    selectedYear=null
) {

    const row =
        byDriver[
            driver
        ];


    const trajectory =
        DATA.trajectories[
            driver
        ];


    if (
        selectedYear === null
        ||
        !trajectory.years.includes(
            Number(
                selectedYear
            )
        )
    ) {

        const peakIndex =
            trajectory.mean.reduce(
                (
                    bestIndex,
                    value,
                    index,
                    values
                ) =>
                    value > values[bestIndex]
                    ? index
                    : bestIndex,
                0
            );


        selectedYear =
            trajectory.years[
                peakIndex
            ];
    }


    selectedYear =
        Number(
            selectedYear
        );


    document.getElementById(
        "driver-cards"
    ).innerHTML = `

        <div class="small-card">

            <div class="small-label">
                Posterior mean JAWS
            </div>

            <div class="small-value">
                ${num(row.jaws_mean)}
            </div>

            <div class="note">
                89%:
                ${num(row.jaws_lb)}
                –
                ${num(row.jaws_ub)}
            </div>

        </div>


        <div class="small-card">

            <div class="small-label">
                Leaderboard rank
            </div>

            <div class="small-value">
                #${row.rank}
            </div>

        </div>


        <div class="small-card">

            <div class="small-label">
                Expected rank
            </div>

            <div class="small-value">
                #${num(row.rank_mean, 1)}
            </div>

            <div class="note">
                median #${num(row.rank_median, 0)}
            </div>

        </div>


        <div class="small-card">

            <div class="small-label">
                89% rank interval — LB
            </div>

            <div class="small-value">
                #${num(row.rank_lb, 0)}
            </div>

            <div class="note">
                5.5th percentile
            </div>

        </div>


        <div class="small-card">

            <div class="small-label">
                89% rank interval — UB
            </div>

            <div class="small-value">
                #${num(row.rank_ub, 0)}
            </div>

            <div class="note">
                94.5th percentile
            </div>

        </div>


        <div class="small-card">

            <div class="small-label">
                Five-season peak
            </div>

            <div class="small-value">
                ${num(row.peak_score)}
            </div>

            <div class="note">
                raw α =
                ${num(row.peak_alpha, 3)}
            </div>

        </div>


        <div class="small-card">

            <div class="small-label">
                Longevity
            </div>

            <div class="small-value">
                ${num(row.longevity_score)}
            </div>

            <div class="note">
                surplus =
                ${num(row.career_surplus, 2)}
            </div>

        </div>


        <div class="small-card">

            <div class="small-label">
                Career sample
            </div>

            <div class="small-value">
                ${row.starts} starts
            </div>

            <div class="note">
                ${row.seasons} modelled seasons
                •
                ${row.unique_teammates} teammates
            </div>

        </div>

    `;


    const lower = {

        x:
            trajectory.years,

        y:
            trajectory.lb,

        type:
            "scatter",

        mode:
            "lines",

        line: {
            width:
                0
        },

        hoverinfo:
            "skip",

        showlegend:
            false

    };


    const upper = {

        x:
            trajectory.years,

        y:
            trajectory.ub,

        type:
            "scatter",

        mode:
            "lines",

        fill:
            "tonexty",

        fillcolor:
            "rgba(88,166,255,0.15)",

        line: {
            width:
                0
        },

        name:
            "89% posterior interval",

        hoverinfo:
            "skip"

    };


    const mean = {

        x:
            trajectory.years,

        y:
            trajectory.mean,

        type:
            "scatter",

        mode:
            "lines+markers",

        name:
            "Posterior mean α",

        line: {
            width:
                3,
            color:
                "#58a6ff"
        },

        marker: {
            size:
                8,
            color:
                "#58a6ff"
        },

        customdata:
            trajectory.p_gt_zero,

        hovertemplate:
            "<b>%{x}</b><br>"
            + "α %{y:.3f}<br>"
            + "P(α > 0) %{customdata:.1%}<br>"
            + "<b>Click for season detail</b>"
            + "<extra></extra>"

    };


    const selectedIndex =
        trajectory.years.indexOf(
            selectedYear
        );


    const selectedMarker = {

        x:
            [
                selectedYear
            ],

        y:
            [
                trajectory.mean[
                    selectedIndex
                ]
            ],

        type:
            "scatter",

        mode:
            "markers",

        name:
            "Selected season",

        marker: {
            size:
                15,
            symbol:
                "diamond-open",
            color:
                "#d29922",
            line: {
                width:
                    3,
                color:
                    "#d29922"
            }
        },

        hoverinfo:
            "skip"

    };


    let trajectoryLayout =
        layoutBase();


    trajectoryLayout.title = {
        text:
            `${row.display_name}: season-level realised ability`,
        x:
            0.02
    };


    trajectoryLayout.xaxis = {
        ...trajectoryLayout.xaxis,
        title:
            "Season",
        dtick:
            Math.max(
                1,
                Math.ceil(
                    trajectory.years.length
                    /
                    10
                )
            )
    };


    trajectoryLayout.yaxis = {
        ...trajectoryLayout.yaxis,
        title:
            "Latent driver ability α"
    };


    trajectoryLayout.shapes = [
        {
            type:
                "line",
            x0:
                Math.min(
                    ...trajectory.years
                ),
            x1:
                Math.max(
                    ...trajectory.years
                ),
            y0:
                0,
            y1:
                0,
            line: {
                color:
                    "#8b949e",
                dash:
                    "dot"
            }
        }
    ];


    Plotly.react(
        "trajectory",
        [
            lower,
            upper,
            mean,
            selectedMarker
        ],
        trajectoryLayout,
        plotConfig
    );


    const trajectoryElement =
        document.getElementById(
            "trajectory"
        );


    if (
        trajectoryElement.removeAllListeners
    ) {

        trajectoryElement.removeAllListeners(
            "plotly_click"
        );
    }


    trajectoryElement.on(
        "plotly_click",
        event => {

            const clickedYear =
                Number(
                    event.points[
                        0
                    ].x
                );

            if (
                trajectory.years.includes(
                    clickedYear
                )
            ) {

                renderDriver(
                    driver,
                    clickedYear
                );
            }
        }
    );


    renderSeasonDetail(
        driver,
        selectedYear
    );


    const probs =
        DATA.rank_distributions[
            driver
        ];


    const ranks =
        probs.map(
            (
                _,
                i
            ) => (
                i + 1
            )
        );


    const rankTrace = {

        x:
            ranks,

        y:
            probs,

        type:
            "bar",

        marker: {
            color:
                "#bc8cff"
        },

        hovertemplate:
            "Rank #%{x}<br>"
            + "Probability %{y:.1%}"
            + "<extra></extra>"

    };


    let rankLayout =
        layoutBase();


    rankLayout.title = {
        text:
            `${row.display_name}: posterior rank distribution`,
        x:
            0.02
    };


    const showFullRankRange =
        document.getElementById(
            "rank-full-range"
        )?.checked
        ?? false;


    const rankRangeLow =
        showFullRankRange
        ? 0.5
        : Math.max(
            0.5,
            row.rank_lb - 3.5
        );


    const rankRangeHigh =
        showFullRankRange
        ? DATA.metadata.n_drivers + 0.5
        : Math.min(
            DATA.metadata.n_drivers + 0.5,
            row.rank_ub + 3.5
        );


    rankLayout.xaxis = {
        ...rankLayout.xaxis,
        title:
            "Pantheon rank",
        range: [
            rankRangeLow,
            rankRangeHigh
        ],
        dtick:
            showFullRankRange
            ? 5
            : (
                rankRangeHigh - rankRangeLow > 18
                ? 2
                : 1
            )
    };


    rankLayout.yaxis = {
        ...rankLayout.yaxis,
        title:
            "Posterior probability",
        tickformat:
            ".0%"
    };


    rankLayout.shapes = [
        {
            type:
                "rect",
            x0:
                row.rank_lb - 0.5,
            x1:
                row.rank_ub + 0.5,
            y0:
                0,
            y1:
                1,
            yref:
                "paper",
            fillcolor:
                "rgba(88,166,255,0.08)",
            line: {
                width:
                    0
            },
            layer:
                "below"
        }
    ];


    rankLayout.annotations = [
        {
            x:
                (
                    row.rank_lb
                    +
                    row.rank_ub
                )
                /
                2,
            y:
                1,
            yref:
                "paper",
            text:
                `89% rank interval: #${num(
                    row.rank_lb,
                    0
                )}–#${num(
                    row.rank_ub,
                    0
                )}`,
            showarrow:
                false,
            yshift:
                15,
            font: {
                color:
                    "#8b949e",
                size:
                    11
            }
        }
    ];


    Plotly.react(
        "rank-dist",
        [
            rankTrace
        ],
        rankLayout,
        plotConfig
    );


    if (
        networkNodeByDriver[
            driver
        ]
        &&
        currentNetworkDriver
        !==
        driver
    ) {

        currentNetworkDriver =
            driver;


        const networkSelect =
            document.getElementById(
                "network-driver-select"
            );


        if (
            networkSelect
        ) {

            networkSelect.value =
                driver;
        }


        if (
            networkReady
        ) {

            renderNetwork();
        }
    }

}


function populateNetworkDriverSelect(
    selectedDriver=null
) {

    const element =
        document.getElementById(
            "network-driver-select"
        );


    const sortedNodes = [
        ...networkNodes
    ].sort(
        (
            a,
            b
        ) =>
            a.display_name.localeCompare(
                b.display_name
            )
    );


    element.innerHTML =
        "";


    sortedNodes.forEach(
        node => {

            const option =
                document.createElement(
                    "option"
                );


            option.value =
                node.driver;


            option.textContent =
                (
                    node.pantheon_rank
                    !== null
                    &&
                    node.pantheon_rank
                    !== undefined
                )
                ? `#${node.pantheon_rank} ${node.display_name}`
                : node.display_name;


            if (
                node.driver
                === selectedDriver
            ) {

                option.selected =
                    true;
            }


            element.appendChild(
                option
            );

        }
    );

}


function networkEdgeForDriver(
    edge,
    driver
) {

    if (
        edge.a
        === driver
    ) {

        return {
            teammate:
                edge.b,
            delta_mean:
                edge.delta_a_minus_b_mean,
            delta_lb:
                edge.delta_a_minus_b_lb,
            delta_ub:
                edge.delta_a_minus_b_ub,
            p_driver_better:
                edge.p_a_better
        };
    }


    if (
        edge.b
        === driver
    ) {

        return {
            teammate:
                edge.a,
            delta_mean:
                (
                    edge.delta_a_minus_b_mean
                    === null
                    ? null
                    : -edge.delta_a_minus_b_mean
                ),
            delta_lb:
                (
                    edge.delta_a_minus_b_ub
                    === null
                    ? null
                    : -edge.delta_a_minus_b_ub
                ),
            delta_ub:
                (
                    edge.delta_a_minus_b_lb
                    === null
                    ? null
                    : -edge.delta_a_minus_b_lb
                ),
            p_driver_better:
                (
                    edge.p_a_better
                    === null
                    ? null
                    : 1 - edge.p_a_better
                )
        };
    }


    return null;
}


function renderNetworkDetail(
    driver
) {

    const node =
        networkNodeByDriver[
            driver
        ];


    if (
        !node
    ) {

        return;
    }


    const minStarts =
        Number(
            document.getElementById(
                "network-min-starts"
            ).value
        );


    const showAllTeammates =
        document.getElementById(
            "network-show-all-teammates"
        )?.checked
        ?? false;


    const connections =
        networkEdges
        .filter(
            edge =>
                (
                    edge.a === driver
                    ||
                    edge.b === driver
                )
                &&
                (
                    showAllTeammates
                    ||
                    edge.shared_starts >= minStarts
                )
        )
        .map(
            edge => ({
                ...edge,
                oriented:
                    networkEdgeForDriver(
                        edge,
                        driver
                    )
            })
        )
        .sort(
            (
                a,
                b
            ) =>
                b.shared_starts
                -
                a.shared_starts
        );


    const pantheonText =
        (
            node.pantheon_rank
            !== null
            &&
            node.pantheon_rank
            !== undefined
        )
        ? `#${node.pantheon_rank} · JAWS ${num(
            node.jaws_mean,
            1
        )}`
        : "Not Pantheon eligible";


    const rows =
        connections.map(
            edge => {

                const teammate =
                    networkNodeByDriver[
                        edge.oriented.teammate
                    ];


                const delta =
                    (
                        edge.oriented.delta_mean
                        === null
                        ||
                        edge.oriented.delta_mean
                        === undefined
                    )
                    ? "—"
                    : (
                        `${signedNum(
                            edge.oriented.delta_mean,
                            3
                        )}`
                        + ` <span class="note">`
                        + `[${intervalText(
                            edge.oriented.delta_lb,
                            edge.oriented.delta_ub,
                            3
                        )}]`
                        + `</span>`
                    );


                const probability =
                    (
                        edge.oriented.p_driver_better
                        === null
                        ||
                        edge.oriented.p_driver_better
                        === undefined
                    )
                    ? "—"
                    : pct(
                        edge.oriented.p_driver_better
                    );


                const seasons =
                    edge.shared_seasons.length
                    === 1
                    ? `${edge.shared_seasons[0]}`
                    : `${edge.first_year}–${edge.last_year}`;


                return `

                    <tr
                        data-network-driver="${edge.oriented.teammate}">

                        <td style="text-align:left;">
                            ${teammate.display_name}
                        </td>

                        <td>
                            ${edge.shared_starts}
                        </td>

                        <td>
                            ${seasons}
                        </td>

                        <td>
                            ${delta}
                        </td>

                        <td>
                            ${probability}
                        </td>

                    </tr>

                `;

            }
        ).join(
            ""
        );


    const explorerButton =
        byDriver[
            driver
        ]
        ? `
            <button
                id="network-open-driver"
                class="network-button"
                type="button">
                Open in driver explorer
            </button>
        `
        : "";


    document.getElementById(
        "network-detail"
    ).innerHTML = `

        <h3>
            ${node.display_name}
        </h3>

        <div class="note">
            ${pantheonText}
        </div>


        <div class="network-stat-grid">

            <div class="network-stat">
                <div class="network-stat-label">
                    Career mean α
                </div>
                <div class="network-stat-value">
                    ${signedNum(
                        node.career_mean_alpha,
                        3
                    )}
                </div>
                <div class="note">
                    89% ${intervalText(
                        node.career_alpha_lb,
                        node.career_alpha_ub,
                        3
                    )}
                </div>
            </div>


            <div class="network-stat">
                <div class="network-stat-label">
                    Actual starts
                </div>
                <div class="network-stat-value">
                    ${node.starts}
                </div>
            </div>


            <div class="network-stat">
                <div class="network-stat-label">
                    Modelled career
                </div>
                <div class="network-stat-value">
                    ${node.first_year}–${node.last_year}
                </div>
                <div class="note">
                    ${node.modelled_seasons} seasons
                </div>
            </div>


            <div class="network-stat">
                <div class="network-stat-label">
                    Unique teammates
                </div>
                <div class="network-stat-value">
                    ${node.unique_teammates}
                </div>
            </div>

        </div>


        <div class="note" style="margin-bottom:7px;">
            Direct teammate connections — click a row to move
            through the network. Δα is selected driver minus
            teammate, averaged across shared modelled seasons.
            ${
                showAllTeammates
                ? "All direct teammates are shown."
                : `Table follows the current ${minStarts}+ shared-start edge filter.`
            }
        </div>


        <div class="network-table-wrap">

            <table>

                <thead>
                    <tr>
                        <th style="text-align:left;">Teammate</th>
                        <th>Starts</th>
                        <th>Seasons</th>
                        <th>Δα</th>
                        <th>P(&gt;)</th>
                    </tr>
                </thead>

                <tbody>
                    ${rows}
                </tbody>

            </table>

        </div>


        <div class="network-action-row">

            ${explorerButton}

            <button
                id="network-focus-neighbours"
                class="network-button"
                type="button">
                Show neighbourhood
            </button>

        </div>

    `;


    document.querySelectorAll(
        "#network-detail tbody tr[data-network-driver]"
    ).forEach(
        element => {

            element.addEventListener(
                "click",
                () => {

                    selectNetworkDriver(
                        element.dataset.networkDriver,
                        true
                    );
                }
            );
        }
    );


    const openButton =
        document.getElementById(
            "network-open-driver"
        );


    if (
        openButton
    ) {

        openButton.addEventListener(
            "click",
            () => {

                document.getElementById(
                    "driver-select"
                ).value =
                    driver;


                renderDriver(
                    driver
                );


                document.getElementById(
                    "driver-select"
                ).scrollIntoView({
                    behavior:
                        "smooth",
                    block:
                        "center"
                });
            }
        );
    }


    const focusButton =
        document.getElementById(
            "network-focus-neighbours"
        );


    if (
        focusButton
    ) {

        focusButton.addEventListener(
            "click",
            () => {

                document.getElementById(
                    "network-view-mode"
                ).value =
                    "ego";


                renderNetwork();
            }
        );
    }

}




function teammateEdgeBetween(
    driverA,
    driverB
) {

    return (
        networkEdges.find(
            edge =>
                (
                    edge.a === driverA
                    &&
                    edge.b === driverB
                )
                ||
                (
                    edge.a === driverB
                    &&
                    edge.b === driverA
                )
        )
        ||
        null
    );
}


function gameDisplayName(
    driver
) {

    const node =
        networkNodeByDriver[
            driver
        ];

    return (
        node
        ? node.display_name
        : driver
    );
}


function gameDriverFromInput(
    value
) {

    const clean =
        String(
            value
            ||
            ""
        ).trim();


    if (
        !clean
    ) {

        return null;
    }


    const lower =
        clean.toLocaleLowerCase();


    const exactDisplay =
        networkNodes.find(
            node =>
                node.display_name.toLocaleLowerCase()
                ===
                lower
        );


    if (
        exactDisplay
    ) {

        return exactDisplay.driver;
    }


    const exactRef =
        networkNodes.find(
            node =>
                node.driver.toLocaleLowerCase()
                ===
                lower
        );


    return (
        exactRef
        ? exactRef.driver
        : null
    );
}


function populateGameDriverOptions() {

    const datalist =
        document.getElementById(
            "game-driver-options"
        );


    datalist.innerHTML =
        "";


    [
        ...networkNodes
    ]
    .sort(
        (
            a,
            b
        ) =>
            a.display_name.localeCompare(
                b.display_name
            )
    )
    .forEach(
        node => {

            const option =
                document.createElement(
                    "option"
                );

            option.value =
                node.display_name;

            datalist.appendChild(
                option
            );
        }
    );
}


function gameDateKey() {

    const now =
        new Date();


    return [
        now.getUTCFullYear(),
        String(
            now.getUTCMonth()
            +
            1
        ).padStart(
            2,
            "0"
        ),
        String(
            now.getUTCDate()
        ).padStart(
            2,
            "0"
        )
    ].join(
        "-"
    );
}


function gameHashString(
    text
) {

    let hash =
        2166136261;


    for (
        let i = 0;
        i < text.length;
        i += 1
    ) {

        hash ^=
            text.charCodeAt(
                i
            );

        hash =
            Math.imul(
                hash,
                16777619
            );
    }


    return (
        hash
        >>>
        0
    );
}


function dailyGameChallenge() {

    if (
        !gameChallenges.length
    ) {

        return null;
    }


    const key =
        gameDateKey();


    const index =
        gameHashString(
            "pantheon-teammate-chain-"
            +
            key
        )
        %
        gameChallenges.length;


    return gameChallenges[
        index
    ];
}


function randomGameChallenge(
    tier
) {

    let pool =
        gameChallenges;


    if (
        tier
        &&
        tier !== "Any"
    ) {

        pool =
            gameChallenges.filter(
                challenge =>
                    challenge.difficulty_tier
                    ===
                    tier
            );
    }


    if (
        !pool.length
    ) {

        pool =
            gameChallenges;
    }


    if (
        !pool.length
    ) {

        return null;
    }


    let challenge =
        pool[
            Math.floor(
                Math.random()
                *
                pool.length
            )
        ];


    if (
        currentGameChallenge
        &&
        pool.length > 1
    ) {

        let guard =
            0;

        while (
            challenge.a
            ===
            currentGameChallenge.a
            &&
            challenge.b
            ===
            currentGameChallenge.b
            &&
            guard
            <
            20
        ) {

            challenge =
                pool[
                    Math.floor(
                        Math.random()
                        *
                        pool.length
                    )
                ];

            guard +=
                1;
        }
    }


    return challenge;
}


function gameRouteToPathObject(
    drivers
) {

    if (
        !drivers
        ||
        !drivers.length
    ) {

        return null;
    }


    const edges = [];


    for (
        let i = 0;
        i < drivers.length - 1;
        i += 1
    ) {

        const edge =
            teammateEdgeBetween(
                drivers[
                    i
                ],
                drivers[
                    i
                    +
                    1
                ]
            );


        if (
            !edge
        ) {

            return null;
        }


        edges.push(
            edge
        );
    }


    return {
        drivers:
            [
                ...drivers
            ],
        edges:
            edges,
        n_edges:
            edges.length,
        total_shared_starts:
            edges.reduce(
                (
                    total,
                    edge
                ) =>
                    total
                    +
                    edge.shared_starts,
                0
            ),
        bottleneck_shared_starts:
            edges.length
            ? Math.min(
                ...edges.map(
                    edge =>
                        edge.shared_starts
                )
            )
            : null
    };
}


function gameOptimalPath() {

    if (
        !currentGameChallenge
    ) {

        return null;
    }


    return shortestNetworkPath(
        currentGameChallenge.a,
        currentGameChallenge.b,
        1
    );
}


function gameTierMultiplier(
    tier
) {

    const multipliers = {
        Easy:
            1.00,
        Medium:
            1.20,
        Hard:
            1.45,
        Expert:
            1.75,
        Nightmare:
            2.10
    };


    return (
        multipliers[
            tier
        ]
        ||
        1.0
    );
}


function gameScoreValue() {

    if (
        !currentGameChallenge
        ||
        !gameSolved
    ) {

        return null;
    }


    const optimal =
        currentGameChallenge
        .optimal_edges;


    const used =
        Math.max(
            currentGameRoute.length
            -
            1,
            1
        );


    const efficiency =
        optimal
        /
        used;


    const base =
        1000
        *
        gameTierMultiplier(
            currentGameChallenge
            .difficulty_tier
        )
        *
        efficiency;


    return Math.max(
        0,
        Math.round(
            base
            -
            40
            *
            gameInvalidGuesses
            -
            120
            *
            gameHintsUsed
        )
    );
}


function saveGameBest(
    score
) {

    if (
        score === null
        ||
        !currentGameChallenge
    ) {

        return;
    }


    try {

        const tier =
            currentGameChallenge
            .difficulty_tier;


        const tierKey =
            "pantheon-teammate-chain-best-"
            +
            tier;


        const oldTier =
            Number(
                localStorage.getItem(
                    tierKey
                )
                ||
                0
            );


        if (
            score
            >
            oldTier
        ) {

            localStorage.setItem(
                tierKey,
                String(
                    score
                )
            );
        }


        if (
            document.getElementById(
                "game-mode"
            ).value
            ===
            "daily"
        ) {

            const dailyKey =
                "pantheon-teammate-chain-daily-"
                +
                gameDateKey();


            const oldDaily =
                Number(
                    localStorage.getItem(
                        dailyKey
                    )
                    ||
                    0
                );


            if (
                score
                >
                oldDaily
            ) {

                localStorage.setItem(
                    dailyKey,
                    String(
                        score
                    )
                );
            }
        }
    }

    catch (
        error
    ) {

        // localStorage may be unavailable in restrictive browser modes.
    }
}


function renderGameLocalBest() {

    const element =
        document.getElementById(
            "game-local-best"
        );


    if (
        !currentGameChallenge
    ) {

        element.textContent =
            "";

        return;
    }


    let text =
        "";


    try {

        const tier =
            currentGameChallenge
            .difficulty_tier;


        const tierBest =
            Number(
                localStorage.getItem(
                    "pantheon-teammate-chain-best-"
                    +
                    tier
                )
                ||
                0
            );


        if (
            document.getElementById(
                "game-mode"
            ).value
            ===
            "daily"
        ) {

            const dailyBest =
                Number(
                    localStorage.getItem(
                        "pantheon-teammate-chain-daily-"
                        +
                        gameDateKey()
                    )
                    ||
                    0
                );


            text =
                dailyBest
                ? `Today's best: ${dailyBest.toLocaleString()} points`
                : "No score saved for today's challenge yet.";
        }

        else {

            text =
                tierBest
                ? `Your ${tier} best: ${tierBest.toLocaleString()} points`
                : `No saved ${tier} score yet.`;
        }
    }

    catch (
        error
    ) {

        text =
            "Personal bests are stored locally in your browser when available.";
    }


    element.textContent =
        text;
}


function renderGameRoute() {

    const routeElement =
        document.getElementById(
            "game-route"
        );


    if (
        !currentGameRoute.length
    ) {

        routeElement.innerHTML =
            "";

        return;
    }


    const parts = [];


    currentGameRoute.forEach(
        (
            driver,
            index
        ) => {

            parts.push(
                `<span class="game-route-node">`
                +
                `${gameDisplayName(driver)}`
                +
                `</span>`
            );


            if (
                index
                <
                currentGameRoute.length
                -
                1
            ) {

                const edge =
                    teammateEdgeBetween(
                        driver,
                        currentGameRoute[
                            index
                            +
                            1
                        ]
                    );


                const label =
                    edge
                    ? `${edge.shared_starts} shared starts`
                    : "teammates";


                parts.push(
                    `<span class="game-route-edge">`
                    +
                    `→ ${label} →`
                    +
                    `</span>`
                );
            }
        }
    );


    routeElement.innerHTML =
        parts.join(
            ""
        );
}


function renderGameStatus(
    message=null,
    kind=null
) {

    const element =
        document.getElementById(
            "game-status"
        );


    element.className =
        "game-status"
        +
        (
            kind
            ? ` ${kind}`
            : ""
        );


    if (
        message !== null
    ) {

        gameLastMessage =
            {
                message:
                    message,
                kind:
                    kind
            };
    }


    if (
        gameLastMessage
    ) {

        element.innerHTML =
            gameLastMessage.message;

        return;
    }


    if (
        !currentGameChallenge
    ) {

        element.textContent =
            "No challenge loaded.";

        return;
    }


    const current =
        currentGameRoute[
            currentGameRoute.length
            -
            1
        ];


    element.innerHTML =
        `Current driver: <strong>${gameDisplayName(current)}</strong>. `
        +
        `Choose one of their real F1 teammates to continue the chain.`;
}


function renderGameCompletion() {

    const box =
        document.getElementById(
            "game-complete"
        );


    if (
        !currentGameChallenge
        ||
        (
            !gameSolved
            &&
            !gameGaveUp
        )
    ) {

        box.classList.remove(
            "visible"
        );

        return;
    }


    box.classList.add(
        "visible"
    );


    const optimalPath =
        gameOptimalPath();


    const optimalRouteText =
        optimalPath
        ? optimalPath.drivers.map(
            gameDisplayName
        ).join(
            " → "
        )
        : "No optimal route available.";


    document.getElementById(
        "game-optimal-route"
    ).innerHTML =
        `<strong>One shortest route:</strong> `
        +
        optimalRouteText;


    if (
        gameSolved
    ) {

        const score =
            gameScoreValue();


        const used =
            currentGameRoute.length
            -
            1;


        const optimal =
            currentGameChallenge
            .optimal_edges;


        const efficiency =
            Math.round(
                100
                *
                optimal
                /
                Math.max(
                    used,
                    1
                )
            );


        document.getElementById(
            "game-score"
        ).textContent =
            `${score.toLocaleString()} points`;


        document.getElementById(
            "game-result-summary"
        ).innerHTML =
            `<strong>Solved in ${used} links.</strong> `
            +
            `Optimal is ${optimal}. `
            +
            `Route efficiency: ${efficiency}%. `
            +
            `${gameInvalidGuesses} invalid guesses · `
            +
            `${gameHintsUsed} hints.`;


        saveGameBest(
            score
        );
    }

    else {

        document.getElementById(
            "game-score"
        ).textContent =
            "Challenge revealed";


        document.getElementById(
            "game-result-summary"
        ).textContent =
            `Optimal chain length: `
            +
            `${currentGameChallenge.optimal_edges} links.`;
    }


    renderGameLocalBest();
}


function renderTeammateGame() {

    if (
        !currentGameChallenge
    ) {

        return;
    }


    const mode =
        document.getElementById(
            "game-mode"
        ).value;


    const difficultySelect =
        document.getElementById(
            "game-difficulty"
        );


    difficultySelect.disabled =
        mode
        ===
        "daily";


    document.getElementById(
        "game-start-name"
    ).textContent =
        gameDisplayName(
            currentGameChallenge.a
        );


    document.getElementById(
        "game-target-name"
    ).textContent =
        gameDisplayName(
            currentGameChallenge.b
        );


    const badge =
        document.getElementById(
            "game-difficulty-badge"
        );


    badge.dataset.tier =
        currentGameChallenge
        .difficulty_tier;


    badge.textContent =
        `${currentGameChallenge.difficulty_tier} · `
        +
        `${Math.round(currentGameChallenge.difficulty_score)}/100`;


    const dailyLabel =
        document.getElementById(
            "game-daily-label"
        );


    dailyLabel.textContent =
        mode === "daily"
        ? `Daily · ${gameDateKey()}`
        : "Random";


    document.getElementById(
        "game-links-used"
    ).textContent =
        String(
            Math.max(
                currentGameRoute.length
                -
                1,
                0
            )
        );


    document.getElementById(
        "game-invalid"
    ).textContent =
        String(
            gameInvalidGuesses
        );


    document.getElementById(
        "game-hints"
    ).textContent =
        String(
            gameHintsUsed
        );


    const input =
        document.getElementById(
            "game-next-driver"
        );


    input.disabled =
        (
            gameSolved
            ||
            gameGaveUp
        );


    document.getElementById(
        "game-submit"
    ).disabled =
        (
            gameSolved
            ||
            gameGaveUp
        );


    document.getElementById(
        "game-undo"
    ).disabled =
        (
            currentGameRoute.length
            <=
            1
            ||
            gameSolved
            ||
            gameGaveUp
        );


    document.getElementById(
        "game-hint"
    ).disabled =
        (
            gameSolved
            ||
            gameGaveUp
        );


    document.getElementById(
        "game-give-up"
    ).disabled =
        (
            gameSolved
            ||
            gameGaveUp
        );


    renderGameRoute();

    renderGameStatus();

    renderGameCompletion();

    renderGameLocalBest();
}


function startGameChallenge(
    challenge
) {

    if (
        !challenge
    ) {

        renderGameStatus(
            "No suitable challenge could be generated.",
            "error"
        );

        return;
    }


    currentGameChallenge =
        challenge;

    currentGameRoute = [
        challenge.a
    ];

    gameInvalidGuesses =
        0;

    gameHintsUsed =
        0;

    gameSolved =
        false;

    gameGaveUp =
        false;

    gameLastMessage =
        null;


    document.getElementById(
        "game-next-driver"
    ).value =
        "";


    renderTeammateGame();
}


function loadGameForCurrentMode() {

    const mode =
        document.getElementById(
            "game-mode"
        ).value;


    const difficulty =
        document.getElementById(
            "game-difficulty"
        ).value;


    const challenge =
        mode === "daily"
        ? dailyGameChallenge()
        : randomGameChallenge(
            difficulty
        );


    startGameChallenge(
        challenge
    );
}


function submitGameDriver() {

    if (
        !currentGameChallenge
        ||
        gameSolved
        ||
        gameGaveUp
    ) {

        return;
    }


    const input =
        document.getElementById(
            "game-next-driver"
        );


    const nextDriver =
        gameDriverFromInput(
            input.value
        );


    if (
        !nextDriver
    ) {

        gameInvalidGuesses +=
            1;


        renderGameStatus(
            "Choose a driver from the autocomplete list.",
            "error"
        );

        input.value =
            "";

        renderTeammateGame();

        return;
    }


    const currentDriver =
        currentGameRoute[
            currentGameRoute.length
            -
            1
        ];


    if (
        currentGameRoute.includes(
            nextDriver
        )
    ) {

        gameInvalidGuesses +=
            1;


        renderGameStatus(
            `${gameDisplayName(nextDriver)} is already in your chain. `
            +
            `Loops do not help the score.`,
            "error"
        );

        input.value =
            "";

        renderTeammateGame();

        return;
    }


    const edge =
        teammateEdgeBetween(
            currentDriver,
            nextDriver
        );


    if (
        !edge
    ) {

        gameInvalidGuesses +=
            1;


        renderGameStatus(
            `<strong>${gameDisplayName(currentDriver)}</strong> and `
            +
            `<strong>${gameDisplayName(nextDriver)}</strong> did not `
            +
            `share an actual F1 race start as teammates in the database.`,
            "error"
        );

        input.value =
            "";

        renderTeammateGame();

        return;
    }


    currentGameRoute.push(
        nextDriver
    );


    input.value =
        "";


    if (
        nextDriver
        ===
        currentGameChallenge.b
    ) {

        gameSolved =
            true;


        renderGameStatus(
            `<strong>Chain complete!</strong> `
            +
            `${gameDisplayName(currentGameChallenge.a)} → `
            +
            `${gameDisplayName(currentGameChallenge.b)} solved.`,
            "success"
        );
    }

    else {

        renderGameStatus(
            `<strong>Valid teammate link.</strong> `
            +
            `${gameDisplayName(currentDriver)} and `
            +
            `${gameDisplayName(nextDriver)} shared `
            +
            `${edge.shared_starts} actual race starts.`,
            null
        );
    }


    renderTeammateGame();
}


function undoGameStep() {

    if (
        currentGameRoute.length
        <=
        1
        ||
        gameSolved
        ||
        gameGaveUp
    ) {

        return;
    }


    currentGameRoute.pop();

    gameLastMessage =
        null;


    renderTeammateGame();
}


function restartTeammateGame() {

    if (
        !currentGameChallenge
    ) {

        return;
    }


    startGameChallenge(
        currentGameChallenge
    );
}


function giveGameHint() {

    if (
        !currentGameChallenge
        ||
        gameSolved
        ||
        gameGaveUp
    ) {

        return;
    }


    const currentDriver =
        currentGameRoute[
            currentGameRoute.length
            -
            1
        ];


    const path =
        shortestNetworkPath(
            currentDriver,
            currentGameChallenge.b,
            1
        );


    if (
        !path
        ||
        path.drivers.length
        <
        2
    ) {

        renderGameStatus(
            "No useful hint is available from the current position.",
            "error"
        );

        return;
    }


    gameHintsUsed +=
        1;


    const nextDriver =
        path.drivers[
            1
        ];


    renderGameStatus(
        `Hint: one shortest route from `
        +
        `<strong>${gameDisplayName(currentDriver)}</strong> can continue `
        +
        `through <strong>${gameDisplayName(nextDriver)}</strong>.`,
        "hint"
    );


    renderTeammateGame();
}


function giveUpTeammateGame() {

    if (
        !currentGameChallenge
        ||
        gameSolved
        ||
        gameGaveUp
    ) {

        return;
    }


    gameGaveUp =
        true;


    renderGameStatus(
        "Challenge revealed. The shortest route is shown below.",
        "hint"
    );


    renderTeammateGame();
}


function showGamePathInNetwork(
    useOptimal=false
) {

    if (
        !currentGameChallenge
    ) {

        return;
    }


    let path =
        null;


    if (
        useOptimal
    ) {

        path =
            gameOptimalPath();
    }

    else {

        path =
            gameRouteToPathObject(
                currentGameRoute
            );
    }


    if (
        !path
    ) {

        return;
    }


    currentNetworkPath =
        path;

    currentNetworkPathA =
        path.drivers[
            0
        ];

    currentNetworkPathB =
        path.drivers[
            path.drivers.length
            -
            1
        ];

    currentNetworkDriver =
        path.drivers[
            0
        ];


    document.getElementById(
        "network-view-mode"
    ).value =
        "path";


    document.getElementById(
        "network-min-starts"
    ).value =
        "1";


    const focus =
        document.getElementById(
            "network-driver-select"
        );


    if (
        focus
    ) {

        focus.value =
            currentNetworkDriver;
    }


    renderNetworkPathStatus();

    renderNetwork();


    document.getElementById(
        "network-section"
    ).scrollIntoView(
        {
            behavior:
                "smooth",
            block:
                "start"
        }
    );
}


function initialiseTeammateChainGame() {

    populateGameDriverOptions();


    document.getElementById(
        "game-mode"
    ).addEventListener(
        "change",
        () => {

            loadGameForCurrentMode();
        }
    );


    document.getElementById(
        "game-difficulty"
    ).addEventListener(
        "change",
        () => {

            if (
                document.getElementById(
                    "game-mode"
                ).value
                ===
                "random"
            ) {

                loadGameForCurrentMode();
            }
        }
    );


    document.getElementById(
        "game-new"
    ).addEventListener(
        "click",
        () => {

            loadGameForCurrentMode();
        }
    );


    document.getElementById(
        "game-submit"
    ).addEventListener(
        "click",
        submitGameDriver
    );


    document.getElementById(
        "game-next-driver"
    ).addEventListener(
        "keydown",
        event => {

            if (
                event.key
                ===
                "Enter"
            ) {

                event.preventDefault();

                submitGameDriver();
            }
        }
    );


    document.getElementById(
        "game-undo"
    ).addEventListener(
        "click",
        undoGameStep
    );


    document.getElementById(
        "game-restart"
    ).addEventListener(
        "click",
        restartTeammateGame
    );


    document.getElementById(
        "game-hint"
    ).addEventListener(
        "click",
        giveGameHint
    );


    document.getElementById(
        "game-give-up"
    ).addEventListener(
        "click",
        giveUpTeammateGame
    );


    document.getElementById(
        "game-view-route"
    ).addEventListener(
        "click",
        () => {

            showGamePathInNetwork(
                false
            );
        }
    );


    document.getElementById(
        "game-view-optimal"
    ).addEventListener(
        "click",
        () => {

            showGamePathInNetwork(
                true
            );
        }
    );


    startGameChallenge(
        dailyGameChallenge()
    );
}



function populateNetworkPathSelects(
    driverA=null,
    driverB=null
) {

    const sortedNodes = [
        ...networkNodes
    ].sort(
        (
            a,
            b
        ) =>
            a.display_name.localeCompare(
                b.display_name
            )
    );


    [
        [
            "network-path-a",
            driverA
        ],
        [
            "network-path-b",
            driverB
        ]
    ].forEach(
        ([
            id,
            selectedDriver
        ]) => {

            const element =
                document.getElementById(
                    id
                );

            element.innerHTML =
                "";

            sortedNodes.forEach(
                node => {

                    const option =
                        document.createElement(
                            "option"
                        );

                    option.value =
                        node.driver;

                    option.textContent =
                        (
                            node.pantheon_rank
                            !== null
                            &&
                            node.pantheon_rank
                            !== undefined
                        )
                        ? `#${node.pantheon_rank} ${node.display_name}`
                        : node.display_name;

                    if (
                        node.driver
                        === selectedDriver
                    ) {

                        option.selected =
                            true;
                    }

                    element.appendChild(
                        option
                    );
                }
            );
        }
    );
}


function shortestNetworkPath(
    startDriver,
    endDriver,
    minStarts
) {

    if (
        !startDriver
        ||
        !endDriver
    ) {

        return null;
    }


    if (
        startDriver === endDriver
    ) {

        return {
            drivers: [
                startDriver
            ],
            edges: [],
            n_edges: 0,
            total_shared_starts: 0,
            bottleneck_shared_starts: null
        };
    }


    const allowedEdges =
        networkEdges.filter(
            edge =>
                edge.shared_starts
                >=
                minStarts
        );


    const adjacency =
        new Map();


    allowedEdges.forEach(
        edge => {

            if (
                !adjacency.has(
                    edge.a
                )
            ) {

                adjacency.set(
                    edge.a,
                    []
                );
            }

            if (
                !adjacency.has(
                    edge.b
                )
            ) {

                adjacency.set(
                    edge.b,
                    []
                );
            }

            adjacency.get(
                edge.a
            ).push({
                neighbour:
                    edge.b,
                edge:
                    edge
            });

            adjacency.get(
                edge.b
            ).push({
                neighbour:
                    edge.a,
                edge:
                    edge
            });
        }
    );


    if (
        !adjacency.has(
            startDriver
        )
        ||
        !adjacency.has(
            endDriver
        )
    ) {

        return null;
    }


    // Breadth-first search guarantees the minimum number of
    // teammate edges. Sorting each adjacency list by shared
    // starts makes equal-length ties prefer stronger links.
    adjacency.forEach(
        neighbours => {

            neighbours.sort(
                (
                    a,
                    b
                ) => {

                    const strengthDiff =
                        b.edge.shared_starts
                        -
                        a.edge.shared_starts;

                    if (
                        strengthDiff !== 0
                    ) {

                        return strengthDiff;
                    }

                    return networkNodeByDriver[
                        a.neighbour
                    ].display_name.localeCompare(
                        networkNodeByDriver[
                            b.neighbour
                        ].display_name
                    );
                }
            );
        }
    );


    const queue = [
        startDriver
    ];

    let head =
        0;

    const visited =
        new Set([
            startDriver
        ]);

    const parent =
        new Map();

    const parentEdge =
        new Map();


    while (
        head < queue.length
    ) {

        const driver =
            queue[
                head
            ];

        head +=
            1;


        if (
            driver === endDriver
        ) {

            break;
        }


        const neighbours =
            adjacency.get(
                driver
            )
            ||
            [];


        for (
            const item
            of neighbours
        ) {

            if (
                visited.has(
                    item.neighbour
                )
            ) {

                continue;
            }

            visited.add(
                item.neighbour
            );

            parent.set(
                item.neighbour,
                driver
            );

            parentEdge.set(
                item.neighbour,
                item.edge
            );

            queue.push(
                item.neighbour
            );
        }
    }


    if (
        !visited.has(
            endDriver
        )
    ) {

        return null;
    }


    const drivers = [
        endDriver
    ];

    const edges = [];

    let cursor =
        endDriver;


    while (
        cursor !== startDriver
    ) {

        const edge =
            parentEdge.get(
                cursor
            );

        const previous =
            parent.get(
                cursor
            );

        if (
            !edge
            ||
            !previous
        ) {

            return null;
        }

        edges.push(
            edge
        );

        drivers.push(
            previous
        );

        cursor =
            previous;
    }


    drivers.reverse();

    edges.reverse();


    return {
        drivers:
            drivers,
        edges:
            edges,
        n_edges:
            edges.length,
        total_shared_starts:
            edges.reduce(
                (
                    total,
                    edge
                ) =>
                    total
                    +
                    edge.shared_starts,
                0
            ),
        bottleneck_shared_starts:
            edges.length
            ? Math.min(
                ...edges.map(
                    edge =>
                        edge.shared_starts
                )
            )
            : null
    };
}


function renderNetworkPathStatus(
    message=null
) {

    const element =
        document.getElementById(
            "network-path-status"
        );


    if (
        message !== null
    ) {

        element.innerHTML =
            message;

        return;
    }


    if (
        !currentNetworkPath
    ) {

        element.innerHTML =
            "Choose two drivers to find the minimum number of "
            + "teammate edges connecting them. The path uses the "
            + "current minimum shared-start threshold; ties favour "
            + "stronger shared-start connections.";

        return;
    }


    const path =
        currentNetworkPath;


    const routeParts = [];


    path.drivers.forEach(
        (
            driver,
            index
        ) => {

            const node =
                networkNodeByDriver[
                    driver
                ];

            routeParts.push(
                `<button class="network-path-node" `
                + `type="button" `
                + `data-path-driver="${driver}">`
                + `${node.display_name}`
                + `</button>`
            );

            if (
                index < path.edges.length
            ) {

                routeParts.push(
                    `<span class="network-path-edge-label">`
                    + `— ${path.edges[index].shared_starts} starts →`
                    + `</span>`
                );
            }
        }
    );


    const edgeWord =
        path.n_edges === 1
        ? "edge"
        : "edges";


    const bottleneckText =
        path.n_edges
        ? ` · weakest link ${path.bottleneck_shared_starts} shared starts`
        : "";


    element.innerHTML =
        `<strong>${path.n_edges} ${edgeWord}</strong>`
        + ` · ${path.drivers.length} drivers`
        + bottleneckText
        + `<div class="network-path-route">`
        + routeParts.join("")
        + `</div>`;


    element.querySelectorAll(
        "[data-path-driver]"
    ).forEach(
        button => {

            button.addEventListener(
                "click",
                () => {

                    selectNetworkDriver(
                        button.dataset.pathDriver,
                        false
                    );

                    renderNetworkDetail(
                        button.dataset.pathDriver
                    );
                }
            );
        }
    );
}


function findAndShowNetworkPath(
    switchToPathView=true
) {

    const driverA =
        document.getElementById(
            "network-path-a"
        ).value;

    const driverB =
        document.getElementById(
            "network-path-b"
        ).value;

    const minStarts =
        Number(
            document.getElementById(
                "network-min-starts"
            ).value
        );


    currentNetworkPathA =
        driverA;

    currentNetworkPathB =
        driverB;


    const path =
        shortestNetworkPath(
            driverA,
            driverB,
            minStarts
        );


    if (
        !path
    ) {

        currentNetworkPath =
            null;


        const fallback =
            minStarts > 1
            ? shortestNetworkPath(
                driverA,
                driverB,
                1
            )
            : null;


        const aName =
            networkNodeByDriver[
                driverA
            ].display_name;

        const bName =
            networkNodeByDriver[
                driverB
            ].display_name;


        if (
            fallback
        ) {

            renderNetworkPathStatus(
                `<strong>No path at ${minStarts}+ shared starts.</strong> `
                + `${aName} and ${bName} are connected in the full `
                + `teammate graph; lower the edge threshold and try again.`
            );
        }

        else {

            renderNetworkPathStatus(
                `<strong>No teammate path found.</strong> `
                + `${aName} and ${bName} are in disconnected components `
                + `of the 1982–2025 modelled teammate network.`
            );
        }


        if (
            document.getElementById(
                "network-view-mode"
            ).value === "path"
        ) {

            document.getElementById(
                "network-view-mode"
            ).value =
                "largest";
        }

        renderNetwork();

        return;
    }


    currentNetworkPath =
        path;

    currentNetworkDriver =
        driverA;


    const focusSelect =
        document.getElementById(
            "network-driver-select"
        );

    if (
        focusSelect
    ) {

        focusSelect.value =
            driverA;
    }


    if (
        switchToPathView
    ) {

        document.getElementById(
            "network-view-mode"
        ).value =
            "path";
    }


    renderNetworkPathStatus();

    renderNetwork();
}


function clearNetworkPath() {

    currentNetworkPath =
        null;

    currentNetworkPathA =
        null;

    currentNetworkPathB =
        null;


    if (
        document.getElementById(
            "network-view-mode"
        ).value === "path"
    ) {

        document.getElementById(
            "network-view-mode"
        ).value =
            "largest";
    }


    renderNetworkPathStatus();

    renderNetwork();
}

function largestComponentDriverSet(
    edges
) {

    const adjacency =
        new Map();


    edges.forEach(
        edge => {

            if (
                !adjacency.has(
                    edge.a
                )
            ) {
                adjacency.set(
                    edge.a,
                    new Set()
                );
            }

            if (
                !adjacency.has(
                    edge.b
                )
            ) {
                adjacency.set(
                    edge.b,
                    new Set()
                );
            }

            adjacency.get(
                edge.a
            ).add(
                edge.b
            );

            adjacency.get(
                edge.b
            ).add(
                edge.a
            );
        }
    );


    let best =
        new Set();

    const visited =
        new Set();


    for (
        const start
        of adjacency.keys()
    ) {

        if (
            visited.has(
                start
            )
        ) {
            continue;
        }

        const component =
            new Set();

        const stack = [
            start
        ];

        visited.add(
            start
        );

        while (
            stack.length
        ) {

            const driver =
                stack.pop();

            component.add(
                driver
            );

            for (
                const neighbour
                of adjacency.get(
                    driver
                )
            ) {

                if (
                    !visited.has(
                        neighbour
                    )
                ) {

                    visited.add(
                        neighbour
                    );

                    stack.push(
                        neighbour
                    );
                }
            }
        }

        if (
            component.size > best.size
        ) {

            best =
                component;
        }
    }

    return best;
}



function componentDriverSetForDriver(
    edges,
    startDriver
) {

    if (
        !startDriver
    ) {

        return new Set();
    }


    const adjacency =
        new Map();


    edges.forEach(
        edge => {

            if (
                !adjacency.has(
                    edge.a
                )
            ) {

                adjacency.set(
                    edge.a,
                    []
                );
            }

            if (
                !adjacency.has(
                    edge.b
                )
            ) {

                adjacency.set(
                    edge.b,
                    []
                );
            }

            adjacency.get(
                edge.a
            ).push(
                edge.b
            );

            adjacency.get(
                edge.b
            ).push(
                edge.a
            );
        }
    );


    const component =
        new Set([
            startDriver
        ]);


    if (
        !adjacency.has(
            startDriver
        )
    ) {

        return component;
    }


    const stack = [
        startDriver
    ];


    while (
        stack.length
    ) {

        const driver =
            stack.pop();


        for (
            const neighbour
            of (
                adjacency.get(
                    driver
                )
                ||
                []
            )
        ) {

            if (
                component.has(
                    neighbour
                )
            ) {

                continue;
            }


            component.add(
                neighbour
            );

            stack.push(
                neighbour
            );
        }
    }


    return component;
}


function pathDisplayCoordinates(
    drivers
) {

    const n =
        drivers.length;


    const centre =
        0.5 * (
            n - 1
        );


    const coords = {};


    drivers.forEach(
        (
            driver,
            index
        ) => {

            coords[
                driver
            ] = {

                // Horizontal position is the order along the route.
                x:
                    index - centre,

                // A small alternating depth offset keeps labels and
                // consecutive path segments visually distinct in 3D.
                y:
                    (
                        n <= 2
                        ? 0
                        : (
                            index % 2 === 0
                            ? -0.22
                            : 0.22
                        )
                    ),

                // Retain the meaningful temporal dimension.
                z:
                    networkNodeByDriver[
                        driver
                    ].z
            };
        }
    );


    return coords;
}


function paddedNetworkRange(
    values,
    fraction=0.18,
    minimumPad=0.20
) {

    if (
        !values.length
    ) {

        return null;
    }


    const lo =
        Math.min(
            ...values
        );

    const hi =
        Math.max(
            ...values
        );

    const span =
        Math.max(
            hi - lo,
            1e-9
        );

    const pad =
        Math.max(
            fraction * span,
            minimumPad
        );


    return [
        lo - pad,
        hi + pad
    ];
}


function renderNetwork() {

    const minStarts =
        Number(
            document.getElementById(
                "network-min-starts"
            ).value
        );


    const mode =
        document.getElementById(
            "network-view-mode"
        ).value;


    const pathDisplayCoords =
        (
            mode === "path"
            &&
            currentNetworkPath
        )
        ? pathDisplayCoordinates(
            currentNetworkPath.drivers
        )
        : null;


    let visibleEdges =
        networkEdges.filter(
            edge =>
                edge.shared_starts
                >=
                minStarts
        );


    let visibleDrivers =
        new Set();


    if (
        mode === "path"
        &&
        currentNetworkPath
    ) {

        // Route-only view.
        //
        // The global force layout remains available in Largest
        // component / Entire network / Neighbourhood views. For a
        // shortest path we deliberately reposition route nodes by
        // path order below so legitimate paths that occupy a tiny
        // part of the global x/y layout do not collapse visually.
        //
        // z still has its real interpretation: career midpoint year.
        visibleDrivers =
            new Set(
                currentNetworkPath.drivers
            );

        visibleEdges =
            [
                ...currentNetworkPath.edges
            ];
    }


    else if (
        mode === "ego"
        &&
        currentNetworkDriver
    ) {

        visibleEdges =
            visibleEdges.filter(
                edge =>
                    edge.a === currentNetworkDriver
                    ||
                    edge.b === currentNetworkDriver
            );

        visibleDrivers.add(
            currentNetworkDriver
        );

        visibleEdges.forEach(
            edge => {

                visibleDrivers.add(
                    edge.a
                );

                visibleDrivers.add(
                    edge.b
                );
            }
        );
    }


    else if (
        mode === "largest"
    ) {

        visibleDrivers =
            largestComponentDriverSet(
                visibleEdges
            );

        visibleEdges =
            visibleEdges.filter(
                edge =>
                    visibleDrivers.has(
                        edge.a
                    )
                    &&
                    visibleDrivers.has(
                        edge.b
                    )
            );
    }


    else {

        if (
            minStarts <= 1
        ) {

            visibleDrivers =
                new Set(
                    networkNodes.map(
                        node =>
                            node.driver
                    )
                );
        }

        else {

            visibleEdges.forEach(
                edge => {

                    visibleDrivers.add(
                        edge.a
                    );

                    visibleDrivers.add(
                        edge.b
                    );
                }
            );

            if (
                currentNetworkDriver
                &&
                networkNodeByDriver[
                    currentNetworkDriver
                ]
            ) {

                visibleDrivers.add(
                    currentNetworkDriver
                );
            }
        }
    }


    const visibleNodes =
        networkNodes.filter(
            node =>
                visibleDrivers.has(
                    node.driver
                )
        );


    const edgeBins = [
        {
            label:
                "1–4 shared starts",
            min:
                1,
            max:
                4,
            width:
                1.0,
            opacity:
                0.16
        },
        {
            label:
                "5–9 shared starts",
            min:
                5,
            max:
                9,
            width:
                1.7,
            opacity:
                0.24
        },
        {
            label:
                "10–19 shared starts",
            min:
                10,
            max:
                19,
            width:
                2.6,
            opacity:
                0.34
        },
        {
            label:
                "20+ shared starts",
            min:
                20,
            max:
                Infinity,
            width:
                4.0,
            opacity:
                0.48
        }
    ];


    const traces = [];


    edgeBins.forEach(
        bin => {

            if (
                mode === "path"
            ) {

                return;
            }


            const binEdges =
                visibleEdges.filter(
                    edge =>
                        edge.shared_starts
                        >=
                        bin.min
                        &&
                        edge.shared_starts
                        <=
                        bin.max
                );


            if (
                !binEdges.length
            ) {

                return;
            }


            const x = [];
            const y = [];
            const z = [];


            binEdges.forEach(
                edge => {

                    const a =
                        networkNodeByDriver[
                            edge.a
                        ];

                    const b =
                        networkNodeByDriver[
                            edge.b
                        ];


                    x.push(
                        a.x,
                        b.x,
                        null
                    );

                    y.push(
                        a.y,
                        b.y,
                        null
                    );

                    z.push(
                        a.z,
                        b.z,
                        null
                    );
                }
            );


            const backgroundOpacity =
                mode === "path"
                ? Math.max(
                    0.035,
                    bin.opacity * 0.24
                )
                : bin.opacity;


            const backgroundWidth =
                mode === "path"
                ? Math.max(
                    0.55,
                    bin.width * 0.48
                )
                : bin.width;


            traces.push({
                x:
                    x,
                y:
                    y,
                z:
                    z,
                type:
                    "scatter3d",
                mode:
                    "lines",
                name:
                    bin.label,
                hoverinfo:
                    "skip",
                line: {
                    color:
                        `rgba(139,148,158,${backgroundOpacity})`,
                    width:
                        backgroundWidth
                }
            });
        }
    );


    const pathDriverSet =
        (
            currentNetworkPath
            &&
            currentNetworkPath.drivers
        )
        ? new Set(
            currentNetworkPath.drivers
        )
        : new Set();


    const baseNodes =
        mode === "path"
        ? visibleNodes.filter(
            node =>
                !pathDriverSet.has(
                    node.driver
                )
        )
        : visibleNodes;


    const nodeHover =
        baseNodes.map(
            node => {

                const pantheon =
                    (
                        node.pantheon_rank
                        !== null
                        &&
                        node.pantheon_rank
                        !== undefined
                    )
                    ? `Pantheon #${node.pantheon_rank}<br>`
                    : "Not Pantheon eligible<br>";


                return (
                    `<b>${node.display_name}</b><br>`
                    + pantheon
                    + `Career mean α ${signedNum(
                        node.career_mean_alpha,
                        3
                    )}<br>`
                    + `Modelled ${node.first_year}–${node.last_year}`
                    + ` (${node.modelled_seasons} seasons)<br>`
                    + `Career midpoint ${(0.5 * (node.first_year + node.last_year)).toFixed(1)}<br>`
                    + `${node.starts} actual starts · `
                    + `${node.unique_teammates} teammates<br>`
                    + `<b>Click to inspect connections</b>`
                );
            }
        );


    traces.push({
        x:
            baseNodes.map(
                node =>
                    node.x
            ),
        y:
            baseNodes.map(
                node =>
                    node.y
            ),
        z:
            baseNodes.map(
                node =>
                    node.z
            ),
        type:
            "scatter3d",
        mode:
            mode === "ego"
            ? "markers+text"
            : "markers",
        name:
            "Drivers",
        showlegend:
            false,
        text:
            mode === "ego"
            ? baseNodes.map(
                node =>
                    node.driver === currentNetworkDriver
                    ? ""
                    : node.display_name
            )
            : undefined,
        textposition:
            "top center",
        textfont: {
            size:
                10,
            color:
                "#c9d1d9"
        },
        customdata:
            baseNodes.map(
                node =>
                    node.driver
            ),
        hovertext:
            nodeHover,
        hovertemplate:
            "%{hovertext}<extra></extra>",
        marker: {
            size:
                baseNodes.map(
                    node => {

                        const normalSize =
                            Math.min(
                                16,
                                4.5
                                +
                                0.72
                                *
                                Math.sqrt(
                                    Math.max(
                                        node.starts,
                                        1
                                    )
                                )
                            );

                        return (
                            mode === "path"
                            ? Math.max(
                                3.5,
                                0.70 * normalSize
                            )
                            : normalSize
                        );
                    }
                ),
            color:
                baseNodes.map(
                    node =>
                        node.career_mean_alpha
                ),
            colorscale:
                "RdBu",
            reversescale:
                true,
            cmin:
                -DATA.network.alpha_absmax,
            cmax:
                DATA.network.alpha_absmax,
            opacity:
                mode === "path"
                ? 0.22
                : 0.90,
            line: {
                color:
                    "#0d1117",
                width:
                    mode === "path"
                    ? 0.4
                    : 0.8
            },
            colorbar: {
                title:
                    "Career mean α",
                thickness:
                    14,
                len:
                    0.68
            }
        }
    });


    if (
        currentNetworkPath
        &&
        currentNetworkPath.edges.length
    ) {

        const pathX = [];
        const pathY = [];
        const pathZ = [];


        currentNetworkPath.edges.forEach(
            edge => {

                if (
                    !visibleDrivers.has(
                        edge.a
                    )
                    ||
                    !visibleDrivers.has(
                        edge.b
                    )
                ) {

                    return;
                }

                const a =
                    (
                        mode === "path"
                        &&
                        pathDisplayCoords
                    )
                    ? pathDisplayCoords[
                        edge.a
                    ]
                    : networkNodeByDriver[
                        edge.a
                    ];

                const b =
                    (
                        mode === "path"
                        &&
                        pathDisplayCoords
                    )
                    ? pathDisplayCoords[
                        edge.b
                    ]
                    : networkNodeByDriver[
                        edge.b
                    ];

                pathX.push(
                    a.x,
                    b.x,
                    null
                );

                pathY.push(
                    a.y,
                    b.y,
                    null
                );

                pathZ.push(
                    a.z,
                    b.z,
                    null
                );
            }
        );


        if (
            pathX.length
        ) {

            traces.push({
                x:
                    pathX,
                y:
                    pathY,
                z:
                    pathZ,
                type:
                    "scatter3d",
                mode:
                    "lines",
                name:
                    "Shortest path",
                hoverinfo:
                    "skip",
                line: {
                    color:
                        "#d29922",
                    width:
                        7
                }
            });
        }


        const pathVisibleDrivers =
            currentNetworkPath.drivers.filter(
                driver =>
                    visibleDrivers.has(
                        driver
                    )
            );


        if (
            pathVisibleDrivers.length
        ) {

            const pathLabels =
                pathVisibleDrivers.map(
                    driver =>
                        networkNodeByDriver[
                            driver
                        ].display_name
                );


            const pathTextPositions =
                pathVisibleDrivers.map(
                    (
                        _driver,
                        index
                    ) => {

                        if (
                            index === 0
                        ) {

                            return "top left";
                        }

                        if (
                            index === pathVisibleDrivers.length - 1
                        ) {

                            return "top right";
                        }

                        return (
                            index % 2 === 0
                            ? "bottom center"
                            : "top center"
                        );
                    }
                );


            traces.push({
                x:
                    pathVisibleDrivers.map(
                        driver =>
                            (
                                mode === "path"
                                &&
                                pathDisplayCoords
                            )
                            ? pathDisplayCoords[
                                driver
                            ].x
                            : networkNodeByDriver[
                                driver
                            ].x
                    ),
                y:
                    pathVisibleDrivers.map(
                        driver =>
                            (
                                mode === "path"
                                &&
                                pathDisplayCoords
                            )
                            ? pathDisplayCoords[
                                driver
                            ].y
                            : networkNodeByDriver[
                                driver
                            ].y
                    ),
                z:
                    pathVisibleDrivers.map(
                        driver =>
                            (
                                mode === "path"
                                &&
                                pathDisplayCoords
                            )
                            ? pathDisplayCoords[
                                driver
                            ].z
                            : networkNodeByDriver[
                                driver
                            ].z
                    ),
                type:
                    "scatter3d",
                mode:
                    "markers+text",
                name:
                    "Path drivers",
                customdata:
                    pathVisibleDrivers,
                text:
                    pathLabels,
                textposition:
                    pathTextPositions,
                textfont: {
                    size:
                        13,
                    color:
                        "#e6edf3"
                },
                hovertemplate:
                    "<b>%{text}</b><br>"
                    + "Shortest-path driver"
                    + "<extra></extra>",
                marker: {
                    size:
                        pathVisibleDrivers.map(
                            (
                                _driver,
                                index
                            ) =>
                                (
                                    index === 0
                                    ||
                                    index === pathVisibleDrivers.length - 1
                                )
                                ? 21
                                : 15
                        ),
                    color:
                        pathVisibleDrivers.map(
                            (
                                _driver,
                                index
                            ) =>
                                index === 0
                                ? "#58a6ff"
                                : (
                                    index === pathVisibleDrivers.length - 1
                                    ? "#bc8cff"
                                    : "#d29922"
                                )
                        ),
                    symbol:
                        pathVisibleDrivers.map(
                            (
                                _driver,
                                index
                            ) =>
                                (
                                    index === 0
                                    ||
                                    index === pathVisibleDrivers.length - 1
                                )
                                ? "diamond"
                                : "circle"
                        ),
                    line: {
                        color:
                            "#ffffff",
                        width:
                            2.0
                    }
                }
            });
        }
    }


    const selectedNode =
        networkNodeByDriver[
            currentNetworkDriver
        ];


    if (
        selectedNode
        &&
        visibleDrivers.has(
            currentNetworkDriver
        )
        &&
        mode !== "path"
    ) {

        traces.push({
            x: [
                selectedNode.x
            ],
            y: [
                selectedNode.y
            ],
            z: [
                selectedNode.z
            ],
            type:
                "scatter3d",
            mode:
                "markers+text",
            name:
                "Selected driver",
            customdata: [
                selectedNode.driver
            ],
            text: [
                selectedNode.display_name
            ],
            textposition:
                "top center",
            hovertemplate:
                `<b>${selectedNode.display_name}</b>`
                + "<br>Selected driver"
                + "<extra></extra>",
            marker: {
                size:
                    18,
                color:
                    "#d29922",
                symbol:
                    "circle",
                line: {
                    color:
                        "#ffffff",
                    width:
                        2
                }
            }
        });
    }


    let pathXRange =
        null;

    let pathYRange =
        null;

    let pathZRange =
        null;


    if (
        mode === "path"
        &&
        currentNetworkPath
        &&
        currentNetworkPath.drivers.length
    ) {

        const pathNodes =
            currentNetworkPath.drivers.map(
                driver =>
                    (
                        pathDisplayCoords
                        ? pathDisplayCoords[
                            driver
                        ]
                        : networkNodeByDriver[
                            driver
                        ]
                    )
            );


        pathXRange =
            paddedNetworkRange(
                pathNodes.map(
                    node =>
                        node.x
                ),
                0.18,
                0.65
            );


        pathYRange =
            [
                -0.75,
                0.75
            ];


        pathZRange =
            paddedNetworkRange(
                pathNodes.map(
                    node =>
                        node.z
                ),
                0.08,
                2.5
            );
    }


    const layout = {
        paper_bgcolor:
            "#161b22",
        plot_bgcolor:
            "#161b22",
        font: {
            color:
                "#c9d1d9"
        },
        margin: {
            l:
                0,
            r:
                0,
            t:
                12,
            b:
                0
        },
        legend: {
            x:
                0.01,
            y:
                0.99,
            bgcolor:
                "rgba(13,17,23,0.55)",
            font: {
                size:
                    10
            }
        },
        scene: {
            bgcolor:
                "#161b22",
            aspectmode:
                "manual",
            aspectratio:
                mode === "path"
                ? {
                    x:
                        1.70,
                    y:
                        0.70,
                    z:
                        1.00
                }
                : {
                    x:
                        1.25,
                    y:
                        1.25,
                    z:
                        1.0
                },
            xaxis: {
                visible:
                    false,
                range:
                    pathXRange
                    ||
                    undefined
            },
            yaxis: {
                visible:
                    false,
                range:
                    pathYRange
                    ||
                    undefined
            },
            zaxis: {
                visible:
                    true,
                title:
                    "Career midpoint year",
                dtick:
                    5,
                range:
                    pathZRange
                    ||
                    undefined,
                gridcolor:
                    "#30363d",
                zeroline:
                    false,
                showbackground:
                    false,
                tickfont: {
                    size:
                        10,
                    color:
                        "#8b949e"
                },
                titlefont: {
                    size:
                        11,
                    color:
                        "#8b949e"
                }
            },
            camera:
                mode === "path"
                ? {
                    eye: {
                        x:
                            1.55,
                        y:
                            1.15,
                        z:
                            0.70
                    }
                }
                : {
                    eye: {
                        x:
                            1.45,
                        y:
                            1.45,
                        z:
                            0.85
                    }
                }
        },
        uirevision:
            (
                mode === "path"
                &&
                currentNetworkPath
            )
            ? (
                "pantheon-network-path-"
                + currentNetworkPath.drivers.join(
                    "-"
                )
                + "-"
                + String(
                    minStarts
                )
            )
            : "pantheon-network-3d",
        hoverlabel: {
            bgcolor:
                "#21262d",
            bordercolor:
                "#30363d",
            font: {
                color:
                    "#e6edf3"
            }
        }
    };


    Plotly.react(
        "teammate-network",
        traces,
        layout,
        {
            ...plotConfig,
            scrollZoom:
                true
        }
    );


    const networkElement =
        document.getElementById(
            "teammate-network"
        );


    if (
        networkElement.removeAllListeners
    ) {

        networkElement.removeAllListeners(
            "plotly_click"
        );
    }


    networkElement.on(
        "plotly_click",
        event => {

            if (
                !event.points.length
            ) {

                return;
            }


            const driver =
                event.points[
                    0
                ].customdata;


            if (
                driver
                &&
                networkNodeByDriver[
                    driver
                ]
            ) {

                selectNetworkDriver(
                    driver,
                    true
                );
            }
        }
    );


    networkReady =
        true;


    const viewLabel =
        mode === "ego"
        ? "neighbourhood"
        : (
            mode === "largest"
            ? "largest component"
            : (
                mode === "path"
                ? "shortest path"
                : "entire network"
            )
        );


    document.getElementById(
        "network-status"
    ).textContent =
        `${visibleNodes.length} drivers · `
        + `${visibleEdges.length} edges shown · `
        + `${networkEdges.length} total teammate edges · `
        + viewLabel;


    renderNetworkDetail(
        currentNetworkDriver
    );

    renderNetworkPathStatus();

}


function selectNetworkDriver(
    driver,
    rerender=true
) {

    if (
        !networkNodeByDriver[
            driver
        ]
    ) {

        return;
    }


    currentNetworkDriver =
        driver;


    const select =
        document.getElementById(
            "network-driver-select"
        );


    if (
        select
    ) {

        select.value =
            driver;
    }


    renderNetworkDetail(
        driver
    );


    if (
        rerender
        &&
        networkReady
    ) {

        renderNetwork();
    }

}


function renderPeakLongevity() {

    const trace = {

        x:
            tableData.map(
                row =>
                    row.peak_score
            ),

        y:
            tableData.map(
                row =>
                    row.longevity_score
            ),

        text:
            tableData.map(
                row =>
                    row.display_name
            ),

        customdata:
            tableData.map(
                row => [

                    row.driver,

                    row.rank,

                    row.jaws_mean,

                    row.rank_lb,

                    row.rank_ub

                ]
            ),

        type:
            "scatter",

        mode:
            "markers",

        marker: {

            size:
                tableData.map(
                    row =>
                        Math.max(
                            7,
                            18
                            -
                            0.08
                            * row.rank
                        )
                ),

            color:
                tableData.map(
                    row =>
                        row.jaws_mean
                ),

            colorscale:
                "Viridis",

            showscale:
                true,

            colorbar: {

                title:
                    "JAWS"

            },

            opacity:
                0.85

        },

        hovertemplate:
            "<b>%{text}</b><br>"
            + "Rank #%{customdata[1]}<br>"
            + "89% rank interval "
            + "#%{customdata[3]:.0f}–#%{customdata[4]:.0f}<br>"
            + "JAWS %{customdata[2]:.1f}<br>"
            + "Peak %{x:.1f}<br>"
            + "Longevity %{y:.1f}"
            + "<extra></extra>"

    };


    let layout =
        layoutBase();


    layout.xaxis = {

        ...layout.xaxis,

        title:
            "Peak score"

    };


    layout.yaxis = {

        ...layout.yaxis,

        title:
            "Longevity score"

    };


    Plotly.newPlot(

        "peak-longevity",

        [
            trace
        ],

        layout,

        plotConfig

    );


    document.getElementById(
        "peak-longevity"
    ).on(

        "plotly_click",

        event => {

            const driver =
                event.points[
                    0
                ].customdata[
                    0
                ];


            document.getElementById(
                "driver-select"
            ).value =
                driver;


            renderDriver(
                driver
            );


            document.getElementById(
                "driver-select"
            ).scrollIntoView({

                behavior:
                    "smooth",

                block:
                    "center"

            });

        }

    );

}


function renderComparison() {

    const a =
        document.getElementById(
            "compare-a"
        ).value;


    const b =
        document.getElementById(
            "compare-b"
        ).value;


    const ia =
        driverIndex[
            a
        ];


    const ib =
        driverIndex[
            b
        ];


    const p =
        DATA.pairwise[
            ia
        ][
            ib
        ];


    const aName =
        byDriver[
            a
        ].display_name;


    const bName =
        byDriver[
            b
        ].display_name;


    let interpretation =
        "";


    if (
        p >= 0.95
    ) {

        interpretation =
            "Very strong posterior separation";

    }


    else if (
        p >= 0.80
    ) {

        interpretation =
            "Clear posterior advantage";

    }


    else if (
        p >= 0.65
    ) {

        interpretation =
            "Moderate posterior advantage";

    }


    else {

        interpretation =
            "Substantial posterior overlap";

    }


    document.getElementById(
        "compare-result"
    ).innerHTML = `

        <div class="probability">
            ${pct(p)}
        </div>

        <div>

            P(
            <strong>${aName}</strong>
            &gt;
            <strong>${bName}</strong>
            )

        </div>

        <div
            class="note"
            style="margin-top:10px;">

            ${interpretation}

        </div>

    `;

}


function initialise() {

    initialiseSummary();


    const urlSelection =
        readUrlSelection();


    const defaultDriver =
        (
            urlSelection.driver
            &&
            byDriver[
                urlSelection.driver
            ]
        )
        ? urlSelection.driver
        : tableData[
            0
        ].driver;


    const defaultYears =
        DATA.trajectories[
            defaultDriver
        ].years;


    const defaultTrajectory =
        DATA.trajectories[
            defaultDriver
        ];


    const defaultPeakIndex =
        defaultTrajectory.mean.reduce(
            (
                bestIndex,
                value,
                index,
                values
            ) =>
                value > values[bestIndex]
                ? index
                : bestIndex,
            0
        );


    const defaultYear =
        (
            urlSelection.year
            &&
            defaultYears.includes(
                urlSelection.year
            )
        )
        ? urlSelection.year
        : defaultYears[
            defaultPeakIndex
        ];


    const secondDriver =
        tableData[
            1
        ].driver;


    populateSelect(
        document.getElementById(
            "driver-select"
        ),
        defaultDriver
    );


    populateSelect(
        document.getElementById(
            "compare-a"
        ),
        defaultDriver
    );


    populateSelect(
        document.getElementById(
            "compare-b"
        ),
        secondDriver
    );


    currentNetworkDriver =
        networkNodeByDriver[
            defaultDriver
        ]
        ? defaultDriver
        : networkNodes[
            0
        ].driver;


    populateNetworkDriverSelect(
        currentNetworkDriver
    );


    populateNetworkPathSelects(
        defaultDriver,
        secondDriver
    );


    initialiseTeammateChainGame();


    renderNetworkPathStatus();


    renderLeaderboard();


    renderDriver(
        defaultDriver,
        defaultYear
    );


    renderPeakLongevity();


    renderComparison();


    renderNetwork();


    document.getElementById(
        "search"
    ).addEventListener(
        "input",
        renderLeaderboard
    );


    document.getElementById(
        "sort"
    ).addEventListener(
        "change",
        renderLeaderboard
    );


    document.getElementById(
        "topn"
    ).addEventListener(
        "change",
        renderLeaderboard
    );


    document.getElementById(
        "driver-select"
    ).addEventListener(
        "change",
        event => {

            renderDriver(
                event.target.value
            );

        }
    );


    document.getElementById(
        "rank-full-range"
    ).addEventListener(
        "change",
        () => {

            if (
                currentDriver
            ) {

                renderDriver(
                    currentDriver,
                    currentSeason
                );
            }
        }
    );


    document.getElementById(
        "compare-a"
    ).addEventListener(
        "change",
        renderComparison
    );


    document.getElementById(
        "compare-b"
    ).addEventListener(
        "change",
        renderComparison
    );


    document.getElementById(
        "network-driver-select"
    ).addEventListener(
        "change",
        event => {

            selectNetworkDriver(
                event.target.value,
                true
            );
        }
    );


    document.getElementById(
        "network-min-starts"
    ).addEventListener(
        "change",
        () => {

            if (
                currentNetworkPathA
                &&
                currentNetworkPathB
            ) {

                findAndShowNetworkPath(
                    false
                );
            }

            else {

                renderNetwork();
            }
        }
    );


    document.getElementById(
        "network-view-mode"
    ).addEventListener(
        "change",
        event => {

            if (
                event.target.value === "path"
                &&
                !currentNetworkPath
            ) {

                findAndShowNetworkPath(
                    true
                );

                return;
            }

            renderNetwork();
        }
    );


    document.getElementById(
        "network-find-path"
    ).addEventListener(
        "click",
        () => {

            findAndShowNetworkPath(
                true
            );
        }
    );


    document.getElementById(
        "network-clear-path"
    ).addEventListener(
        "click",
        clearNetworkPath
    );


    document.getElementById(
        "network-show-all-teammates"
    ).addEventListener(
        "change",
        () => {

            renderNetworkDetail(
                currentNetworkDriver
            );
        }
    );


    document.getElementById(
        "network-reset-camera"
    ).addEventListener(
        "click",
        () => {

            Plotly.relayout(
                "teammate-network",
                {
                    "scene.camera": {
                        eye: {
                            x: 1.45,
                            y: 1.45,
                            z: 0.85
                        }
                    }
                }
            );
        }
    );

}


initialise();


</script>

</body>

</html>
"""


# ==========================================================
# 11. INSERT FINAL DATA
# ==========================================================

html = (
    html.replace(
        "__PANTHEON_DATA__",
        payload_json,
    )
    .replace(
        "__START_YEAR__",
        str(START_YEAR),
    )
    .replace(
        "__END_YEAR__",
        str(END_YEAR),
    )
)


# ==========================================================
# 12. WRITE HTML
# ==========================================================

PANTHEON_DASHBOARD_PATH.write_text(
    html,
    encoding="utf-8",
)


print("========================================")

print("FINAL PANTHEON DASHBOARD BUILT")

print("========================================")


print("\nDashboard ->")

print(PANTHEON_DASHBOARD_PATH.resolve())


print("\nDrivers:", len(eligible_drivers))


print("Posterior draws:", eligible_jaws_mat.shape[0])


print("Era:", f"{START_YEAR}–{END_YEAR}")


print("\nSeason explorer:")

print(
    "  clickable driver seasons = enabled"
)

print(
    "  constructor/car context = enabled"
)

print(
    "  teammate comparison = enabled"
)

print(
    "  shareable driver/year URL state = enabled"
)


# ==========================================================
# 13. OPEN IN DEFAULT BROWSER
# ==========================================================

if OPEN_AFTER_BUILD:

    webbrowser.open(PANTHEON_DASHBOARD_PATH.resolve().as_uri())


print("\nStage 20 complete.")