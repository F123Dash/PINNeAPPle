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
| Light GA (C172 class) | 3.48 | 11.09 | 7.35 | 1.77 | 5.05 | 1.90 | 384 |
| Turboprop (ATR 72 class) | 1.01 | 1.25 | 5.68 | 4.45 | 1.76 | 3.95 | 1278 |
| Airliner (A320 class) | 1.51 | 4.99 | 11.62 | 14.90 | 0.19 | 2.67 | 1666 |
| Motor glider (AR 28) | 1.36 | 5.64 | 11.15 | 3.01 | 4.76 | 0.63 | 277 |

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

| aircraft | controls | V [m/s] | W/S [Pa] | A = ρVCLα/2(W/S) [g per m/s] | A·σ_w [g] | RMS Δn_z CG [g] | spread over seeds | max Δn_z | min Δn_z | RMS bank [deg] | peak bank [deg] | RMS pitch [deg] | RMS n_y CG [g] | worst cabin station: RMS / peak Δn_z [g] | max α [deg] | α at clean CLmax [deg] | model validity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | stick fixed | 55 | 666 | 0.1919 | 1.163 | 0.712 | 0.293 | 3.28 | -1.49 | 23.1 | 59.0 | 17.1 | 0.058 | rear seat: 0.734 / 3.42 | 12.7 | 14.1 | beyond stall |
| Light GA (C172 class) | autopilot | 55 | 666 | 0.1919 | 1.380 | 0.355 | 0.062 | 1.20 | -1.23 | 8.4 | 22.5 | 4.6 | 0.033 | rear seat: 0.376 / 1.35 | 17.1 | 14.1 | beyond stall |
| Turboprop (ATR 72 class) | stick fixed | 120 | 3698 | 0.0846 | 0.539 | 0.628 | 0.522 | 2.80 | -0.91 | 19.3 | 49.0 | 10.5 | 0.084 | aft row: 0.652 / 2.95 | 6.7 | 11.8 | valid |
| Turboprop (ATR 72 class) | autopilot | 120 | 3698 | 0.0846 | 0.576 | 0.276 | 0.012 | 0.90 | -0.84 | 4.2 | 11.9 | 3.4 | 0.034 | aft row: 0.299 / 1.06 | 10.1 | 11.8 | valid |
| Airliner (A320 class) | stick fixed | 128 | 5879 | 0.0520 | 0.331 | 0.365 | 0.122 | 1.44 | -0.72 | 15.4 | 40.3 | 10.1 | 0.040 | aft row: 0.401 / 1.66 | 13.5 | 14.1 | beyond stall |
| Airliner (A320 class) | autopilot | 128 | 5879 | 0.0520 | 0.321 | 0.216 | 0.021 | 0.57 | -0.69 | 4.1 | 12.1 | 2.7 | 0.026 | aft row: 0.249 / 0.80 | 14.6 | 14.1 | beyond stall |
| Motor glider (AR 28) | stick fixed | 40 | 446 | 0.2471 | 1.667 | 33.091 | 97.174 | 1743.49 | -15.98 | 39.0 | 98.4 | 20.4 | 1.073 | pilot: 32.332 / 1629.10 | 73.8 | 7.4 | departed |
| Motor glider (AR 28) | autopilot | 40 | 446 | 0.2471 | 1.517 | 0.452 | 0.049 | 1.51 | -1.46 | 11.3 | 34.8 | 7.4 | 0.048 | pilot: 0.445 / 1.51 | 26.5 | 7.4 | departed |

## spanwise

| aircraft | RMS bank, 3-D field [deg] | RMS bank, frozen line [deg] | RMS Δn_z, 3-D field [g] | RMS Δn_z, frozen line [g] |
|---|---|---|---|---|
| Light GA (C172 class) | 23.1 | 12.2 | 0.712 | 0.412 |
| Turboprop (ATR 72 class) | 19.3 | 2.4 | 0.628 | 0.350 |
| Airliner (A320 class) | 15.4 | 8.3 | 0.365 | 0.250 |
| Motor glider (AR 28) | 39.0 | 34.9 | 33.091 | 0.935 |

## intensity

| aircraft | severity | σ [m/s] | RMS Δn_z [g] | max Δn_z [g] | RMS bank [deg] | max α [deg] | model validity |
|---|---|---|---|---|---|---|---|
| Airliner (A320 class) | light | 1.84 | 0.056 | 0.14 | 1.0 | 8.2 | valid |
| Airliner (A320 class) | moderate | 2.88 | 0.084 | 0.24 | 1.5 | 9.4 | valid |
| Airliner (A320 class) | severe (mean of seeds) | 7.05 | 0.216 | 0.57 | 4.1 | 14.6 | beyond stall |
| Light GA (C172 class) | light | 1.84 | 0.100 | 0.30 | 2.1 | 7.1 | valid |
| Light GA (C172 class) | moderate | 2.88 | 0.147 | 0.59 | 2.8 | 8.9 | valid |
| Light GA (C172 class) | severe (mean of seeds) | 7.05 | 0.355 | 1.20 | 8.4 | 17.1 | beyond stall |
| Motor glider (AR 28) | light | 1.84 | 0.118 | 0.42 | 2.4 | 4.5 | valid |
| Motor glider (AR 28) | moderate | 2.88 | 0.185 | 0.58 | 3.5 | 8.6 | beyond stall |
| Motor glider (AR 28) | severe (mean of seeds) | 7.05 | 0.452 | 1.51 | 11.3 | 26.5 | departed |
| Turboprop (ATR 72 class) | light | 1.84 | 0.081 | 0.26 | 1.1 | 3.0 | valid |
| Turboprop (ATR 72 class) | moderate | 2.88 | 0.125 | 0.40 | 1.7 | 3.9 | valid |
| Turboprop (ATR 72 class) | severe (mean of seeds) | 7.05 | 0.276 | 0.90 | 4.2 | 10.1 | valid |

## pratt

| aircraft | V [m/s] | mass ratio μ | K_g | Δn, no alleviation | Δn Pratt | Δn simulated | sim / Pratt |
|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | 55 | 19.7 | 0.693 | 1.698 | 1.177 | 1.283 | 1.090 |
| Turboprop (ATR 72 class) | 120 | 62.9 | 0.812 | 0.748 | 0.607 | 0.673 | 1.109 |
| Airliner (A320 class) | 128 | 62.1 | 0.811 | 0.460 | 0.373 | 0.407 | 1.092 |
| Motor glider (AR 28) | 40 | 19.1 | 0.689 | 2.185 | 1.506 | 1.794 | 1.191 |

## same_gust

| aircraft | V [m/s] | W/S [Pa] | gust duration [s] | peak Δn_z CG [g] | peak cabin Δn_z [g] | quasi-steady A·U [g] | Pratt (for H = 12.5 MAC) [g] | max Δθ [deg] | max Δh [m] |
|---|---|---|---|---|---|---|---|---|---|
| Light GA (C172 class) | 55 | 666 | 1.82 | 0.625 | rear seat: 0.652 | 1.463 | 1.014 | 5.40 | 4.4 |
| Turboprop (ATR 72 class) | 120 | 3698 | 0.83 | 0.571 | aft row: 0.627 | 0.644 | 0.523 | 1.31 | 1.6 |
| Airliner (A320 class) | 128 | 5879 | 0.78 | 0.357 | aft row: 0.429 | 0.396 | 0.321 | 0.86 | 1.4 |
| Motor glider (AR 28) | 40 | 446 | 2.50 | 0.939 | pilot: 0.934 | 1.883 | 1.297 | 3.21 | 15.5 |

