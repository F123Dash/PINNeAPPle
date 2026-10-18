## psd

| generator | form | comp. | σ target [m/s] | σ generated [m/s] | PSD ratio min | PSD ratio max |
|---|---|---|---|---|---|---|
| FrozenLine | dryden | u | 7.050 | 7.038 | 0.99 | 1.01 |
| FrozenLine | dryden | v | 7.050 | 7.042 | 0.99 | 1.01 |
| FrozenLine | dryden | w | 7.050 | 7.042 | 0.99 | 1.01 |
| FrozenLine | von_karman | u | 7.050 | 7.019 | 0.99 | 1.02 |
| FrozenLine | von_karman | v | 7.050 | 7.019 | 0.99 | 1.01 |
| FrozenLine | von_karman | w | 7.050 | 7.019 | 0.99 | 1.04 |
| DrydenFilter | dryden | u | 2.879 | 2.823 | 0.97 | 1.07 |
| DrydenFilter | dryden | v | 2.879 | 2.894 | 0.93 | 1.05 |
| DrydenFilter | dryden | w | 2.879 | 2.795 | 0.96 | 1.06 |
| FrozenField3D | von_karman | u | 6.991 | 6.917 | 0.79 | 1.09 |
| FrozenField3D | von_karman | v | 6.991 | 7.465 | 0.84 | 1.34 |
| FrozenField3D | von_karman | w | 6.991 | 6.341 | 0.66 | 1.18 |

## trim

| aircraft | V [m/s] | h [m] | α [deg] | δe [deg] | throttle | trim residual [m/s²] | 10 s: max |ΔV| [m/s] | max |Δh| [m] | max |Δθ| [deg] |
|---|---|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | 55 | 1500 | 3.00 | -1.61 | 0.74 | 1.44e-14 | 1.00e-09 | 0.0000 | 3.98e-16 |
| Turboprop (ATR 72 class) | 120 | 3000 | 1.45 | -0.83 | 0.70 | 3.80e-15 | 1.00e-09 | 0.0000 | 0.0000 |
| Airliner (A320 class) | 128 | 3000 | 6.84 | -3.56 | 0.30 | 1.12e-14 | 1.00e-09 | 0.0000 | 0.0000 |
| Motor glider (AR 28) | 30 | 1500 | 3.98 | -2.42 | 0.16 | 1.99e-14 | 1.00e-09 | 0.0000 | 1.59e-15 |

## conservation

| Δt [s] | relative energy error (20 s) | relative angular-momentum error | abs(norm(q) - 1) |
|---|---|---|---|
| 0.020 | 4.40e-10 | 2.68e-09 | 0.00 |
| 0.010 | 8.03e-12 | 1.67e-10 | 0.00 |
| 0.005 | 1.62e-12 | 1.05e-11 | 0.00 |

## lattice

| aircraft | α [deg] | CL lattice in the loop | CL steady lattice | relative difference |
|---|---|---|---|---|
| Light GA (C172 class) | 0 | 0.1617 | 0.1617 | 4.00e-11 |
| Light GA (C172 class) | 4 | 0.5194 | 0.5201 | 0.0013 |
| Turboprop (ATR 72 class) | 0 | 0.4257 | 0.4257 | 4.00e-11 |
| Turboprop (ATR 72 class) | 4 | 0.8254 | 0.8268 | 0.0016 |
| Airliner (A320 class) | 0 | 0.1863 | 0.1863 | 4.00e-11 |
| Airliner (A320 class) | 4 | 0.5564 | 0.5572 | 0.0014 |
| Motor glider (AR 28) | 0 | 0.5238 | 0.5238 | 4.00e-11 |
| Motor glider (AR 28) | 4 | 0.9457 | 0.9473 | 0.0017 |

## aircraft

| aircraft | mass [kg] | S [m²] | span [m] | W/S [Pa] | Ixx [kg m²] | Iyy | Izz |
|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | 1100 | 16.2 | 11.02 | 666 | 1890 | 2905 | 4505 |
| Turboprop (ATR 72 class) | 23000 | 61.0 | 27.05 | 3698 | 288155 | 520555 | 754446 |
| Airliner (A320 class) | 73500 | 122.6 | 34.13 | 5879 | 1385361 | 4014283 | 5200376 |
| Motor glider (AR 28) | 850 | 18.7 | 23.00 | 446 | 9076 | 1895 | 10780 |

## derivatives

| aircraft | CLα | Cmα | static margin | Cmq | Clβ | Cnβ | Clp | Cnr | Cmδe | Clδa | Cnδr | oscillatory modes ωn [rad/s] / ζ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | 5.12 | -1.50 | 0.29 | -18.44 | -0.042 | 0.064 | -0.50 | -0.106 | -1.68 | -0.193 | -0.066 | 5.10 / 0.54, 2.14 / 0.20, 0.21 / 0.10 |
| Turboprop (ATR 72 class) | 5.73 | -1.76 | 0.31 | -54.49 | -0.085 | 0.135 | -0.60 | -0.146 | -2.64 | -0.187 | -0.114 | 1.95 / 0.45, 1.45 / 0.09, 0.11 / 0.05 |
| Airliner (A320 class) | 5.26 | -1.94 | 0.37 | -44.59 | -0.217 | 0.080 | -0.47 | -0.246 | -2.38 | -0.072 | -0.129 | 1.44 / 0.40, 1.00 / 0.11, 0.10 / 0.02 |
| Motor glider (AR 28) | 6.02 | -0.32 | 0.05 | -21.49 | -0.083 | 0.044 | -0.69 | -0.028 | -1.13 | -0.345 | -0.042 | 2.00 / 0.84, 1.19 / 0.27, 0.27 / 0.04 |

## takeoff

| aircraft | V_R [kt] | V_LOF [kt] | θ at lift-off [deg] | α at lift-off [deg] | ground roll [m] | to 35 ft [m] | V at 35 ft [kt] | climb gradient [%] | Raymer, V_LOF = 1.1 V_s [m] | Raymer, simulated V_LOF and rotation time [m] | min tail clearance [m] | max CL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | 55.7 | 62.4 | 6.8 | 6.5 | 244 | 403 | 67.9 | 13.6 | 311 | 410 | 0.95 | 1.10 |
| Turboprop (ATR 72 class) | 109.5 | 119.7 | 6.3 | 6.0 | 1060 | 1291 | 123.2 | 7.3 | 1299 | 1492 | 0.23 | 1.81 |
| Airliner (A320 class) | 141.4 | 155.6 | 9.7 | 9.6 | 1346 | 1691 | 168.2 | 10.4 | 1703 | 1882 | 0.18 | 1.57 |
| Motor glider (AR 28) | 47.9 | 54.3 | 1.6 | 1.3 | 186 | 286 | 57.7 | 16.8 | 250 | 326 | 1.17 | 1.07 |

## crosswind

| aircraft | max offset on the ground [m] | max bank [deg] | max rudder [deg] | max aileron [deg] | heading at lift-off [deg] | min wing-tip clearance [m] | to 35 ft [m] |
|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | 4.87 | 2.01 | 6.61 | 1.00 | 6.29 | 1.89 | 382 |
| Turboprop (ATR 72 class) | 1.47 | 0.93 | 5.39 | 4.47 | 1.98 | 3.95 | 1278 |
| Airliner (A320 class) | 5.95 | 3.19 | 11.76 | 14.28 | 0.59 | 2.65 | 1663 |
| Motor glider (AR 28) | 2.69 | 3.03 | 10.96 | 2.49 | 5.55 | 0.63 | 275 |

## gust

| aircraft | to 35 ft [m] | change vs nominal [m] | V_LOF [kt] | θ at lift-off [deg] | min tail clearance [m] |
|---|---|---|---|---|---|
| Light GA (C172 class) | 394 | -9 | 60.8 | 9.2 | 0.71 |
| Turboprop (ATR 72 class) | 1293 | 2 | 119.6 | 6.3 | 0.22 |
| Airliner (A320 class) | 1693 | 1 | 155.6 | 9.8 | 0.18 |
| Motor glider (AR 28) | 284 | -2 | 54.2 | 2.9 | 1.09 |

## a320

| quantity | value |
|---|---|
| simulated all-engines distance to 35 ft, 73500 kg, ISA sea level | 1691 m |
| simulated x 1.15 (the all-engines part of the certified rule) | 1945 m |
| published | A320-200, MTOW 78 t, sea level ISA: take-off field length about 2,090-2,100 m (certified, one engine inoperative and 1.15 x all-engines distance, whichever is longer); e.g. https://www.skytamer.com/Airbus_A320.html, https://aircraftinvestigation.info/airplanes/Airbus_A320-200_CFM56.html (which also quotes about 1,825 m at 73.5 t). Authoritative source: Airbus, A320 Aircraft Characteristics - Airport and Maintenance Planning (rev. Jun 2024). |

## turbulence

| aircraft | controls | V [m/s] | W/S [Pa] | A = ρVCLα/2(W/S) [g per m/s] | A·σ_w [g] | RMS Δn_z CG [g] | spread over seeds | max Δn_z | min Δn_z | RMS bank [deg] | peak bank [deg] | RMS pitch [deg] | RMS n_y CG [g] | worst cabin station: RMS / peak Δn_z [g] | max α [deg] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | stick fixed | 55 | 666 | 0.1919 | 1.077 | 0.493 | 0.000 | 1.71 | -1.27 | 12.5 | 29.4 | 10.7 | 0.049 | rear seat: 0.508 / 1.76 | 11.4 |
| Light GA (C172 class) | autopilot | 55 | 666 | 0.1919 | 0.718 | 0.303 | 0.000 | 1.00 | -0.85 | 6.9 | 15.0 | 2.1 | 0.027 | rear seat: 0.323 / 1.14 | 13.1 |
| Turboprop (ATR 72 class) | stick fixed | 120 | 3698 | 0.0846 | 0.564 | 0.295 | 0.000 | 0.87 | -0.81 | 6.7 | 13.6 | 4.5 | 0.059 | aft row: 0.322 / 0.89 | 7.2 |
| Turboprop (ATR 72 class) | autopilot | 120 | 3698 | 0.0846 | 0.476 | 0.266 | 0.000 | 0.68 | -0.74 | 3.2 | 7.4 | 1.7 | 0.031 | aft row: 0.288 / 0.82 | 9.5 |
| Airliner (A320 class) | stick fixed | 128 | 5879 | 0.0520 | 0.301 | 0.200 | 0.000 | 0.36 | -0.67 | 18.2 | 39.9 | 5.4 | 0.052 | aft row: 0.247 / 0.88 | 12.8 |
| Airliner (A320 class) | autopilot | 128 | 5879 | 0.0520 | 0.301 | 0.162 | 0.000 | 0.39 | -0.37 | 3.7 | 9.9 | 2.2 | 0.022 | aft row: 0.188 / 0.50 | 12.3 |
| Motor glider (AR 28) | stick fixed | 30 | 446 | 0.1832 | 0.897 | 0.661 | 0.000 | 2.47 | -1.01 | 14.2 | 31.6 | 8.5 | 0.085 | pilot: 0.665 / 2.62 | 14.4 |
| Motor glider (AR 28) | autopilot | 30 | 446 | 0.1832 | 0.838 | 0.412 | 0.000 | 1.09 | -1.52 | 8.6 | 21.0 | 6.6 | 0.061 | pilot: 0.402 / 1.49 | 20.0 |

## spanwise

| aircraft | RMS bank, 3-D field [deg] | RMS bank, frozen line [deg] | RMS Δn_z, 3-D field [g] | RMS Δn_z, frozen line [g] |
|---|---|---|---|---|
| Light GA (C172 class) | 12.5 | 5.5 | 0.493 | 0.477 |
| Turboprop (ATR 72 class) | 6.7 | 2.5 | 0.295 | 0.444 |
| Airliner (A320 class) | 18.2 | 8.8 | 0.200 | 0.254 |
| Motor glider (AR 28) | 14.2 | 3.4 | 0.661 | 0.373 |

## intensity

| aircraft | severity | σ [m/s] | RMS Δn_z [g] | max Δn_z [g] | RMS bank [deg] |
|---|---|---|---|---|---|
| Airliner (A320 class) | light | 1.84 | 0.052 | 0.13 | 1.0 |
| Airliner (A320 class) | moderate | 2.88 | 0.076 | 0.20 | 1.6 |
| Airliner (A320 class) | severe (mean of seeds) | 7.05 | 0.162 | 0.39 | 3.7 |
| Light GA (C172 class) | light | 1.84 | 0.086 | 0.23 | 1.8 |
| Light GA (C172 class) | moderate | 2.88 | 0.118 | 0.30 | 2.0 |
| Light GA (C172 class) | severe (mean of seeds) | 7.05 | 0.303 | 1.00 | 6.9 |
| Motor glider (AR 28) | light | 1.84 | 0.103 | 0.27 | 2.0 |
| Motor glider (AR 28) | moderate | 2.88 | 0.166 | 0.42 | 3.0 |
| Motor glider (AR 28) | severe (mean of seeds) | 7.05 | 0.412 | 1.09 | 8.6 |
| Turboprop (ATR 72 class) | light | 1.84 | 0.082 | 0.18 | 0.9 |
| Turboprop (ATR 72 class) | moderate | 2.88 | 0.123 | 0.25 | 1.3 |
| Turboprop (ATR 72 class) | severe (mean of seeds) | 7.05 | 0.266 | 0.68 | 3.2 |

## pratt

| aircraft | V [m/s] | mass ratio μ | K_g | Δn, no alleviation | Δn Pratt | Δn simulated | sim / Pratt |
|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | 55 | 19.7 | 0.693 | 1.698 | 1.177 | 1.283 | 1.090 |
| Turboprop (ATR 72 class) | 120 | 62.9 | 0.812 | 0.748 | 0.607 | 0.673 | 1.109 |
| Airliner (A320 class) | 128 | 62.1 | 0.811 | 0.460 | 0.373 | 0.407 | 1.092 |
| Motor glider (AR 28) | 30 | 19.3 | 0.691 | 1.621 | 1.120 | 1.331 | 1.189 |

## same_gust

| aircraft | V [m/s] | W/S [Pa] | gust duration [s] | peak Δn_z CG [g] | peak cabin Δn_z [g] | quasi-steady A·U [g] | Pratt (for H = 12.5 MAC) [g] | max Δθ [deg] | max Δh [m] |
|---|---|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | 55 | 666 | 1.82 | 0.625 | rear seat: 0.652 | 1.463 | 1.014 | 5.40 | 4.4 |
| Turboprop (ATR 72 class) | 120 | 3698 | 0.83 | 0.571 | aft row: 0.627 | 0.644 | 0.523 | 1.31 | 1.6 |
| Airliner (A320 class) | 128 | 5879 | 0.78 | 0.357 | aft row: 0.429 | 0.396 | 0.321 | 0.86 | 1.4 |
| Motor glider (AR 28) | 30 | 446 | 3.33 | 0.735 | pilot: 0.733 | 1.396 | 0.965 | 5.69 | 20.7 |

