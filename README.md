# Tello ROS2 Node — Controller & Sensor Diagnostics

Questo repository contiene tre package ROS2 per il drone **DJI Tello**, con dati dal sistema **Vicon Mocap** e una policy di Reinforcement Learning (posizionamento) addestrata in Isaac Lab:

- **`tello_pkg`** — pipeline modulare di controllo, controllata da terminale (`mission_console`), + i checkpoint della policy condivisi (`policy_pos_controller/`).
- **`tello_pkg_web`** — stessa pipeline, controllata da una dashboard web 3D (FastAPI + WebSocket + Three.js) invece che da terminale.
- **`tello_test`** — nodi standalone di diagnostica hardware (`read_sensors`, `takeoff_land`), usati per validare Vicon/telemetria PRIMA di fidarsi del controllore vero.

> Per una versione più estesa (tabelle complete di topic/servizi/frequenze), vedi anche `architettura_ros2.pdf` nella cartella principale del progetto.

---

## 🚀 Guida rapida all'avvio (Docker)

### 1. Avviare il container Docker
Dalla cartella principale del progetto sull'host:
```bash
./run.sh
```
Per aprire una seconda shell nel container attivo:
```bash
./exec.sh
```

### 2. Compilazione del workspace ROS2
All'interno della shell del container (`/ros_workspace`):
```bash
colcon build --symlink-install
source install/setup.bash
```

---

## 🏗️ Architettura: la pipeline modulare

`tello_pkg` è organizzato in **quattro responsabilità separate**, ciascuna un nodo ROS2 indipendente, che comunicano solo tramite topic/servizi (nessuna memoria condivisa):

```
target_handler ──targets──▶ observation_handler ──observations──▶ policy_handler ──policy_action──▶ vel_command_handler
     ▲                              │                                                                        │
     │                    (legge Vicon + targets)                                                djitellopy → drone reale
     └──────────────────── flight_state ─────────────────────────────────────────────────────────────────────┘
```

| Nodo | Package | Responsabilità | Tocca djitellopy? |
| :--- | :--- | :--- | :---: |
| `target_handler` | `tello_pkg` | Genera/avanza la coda target (waypoint). | ❌ |
| `observation_handler` | `tello_pkg` | Legge la posa Vicon, calcola le 52 osservazioni per la policy. | ❌ |
| `policy_handler` | `tello_pkg` | Inferenza della rete RL (checkpoint `policy_pos_controller/`). | ❌ |
| `vel_command_handler` | `tello_pkg` | **Unica** connessione djitellopy reale: takeoff/land/comandi rc/telemetria/failsafe. | ✅ |
| `mission_console` | `tello_pkg` | Controllo interattivo da un **secondo terminale** (wizard parametri, start/land/avanzamento). | ❌ |
| `vel_command_handler_web` | `tello_pkg_web` | Equivalente web di `vel_command_handler` + `mission_console` fusi: stessa attuazione/telemetria + dashboard 3D via HTTP/WebSocket (nessun secondo terminale necessario). | ✅ |

I nodi `target_handler`/`observation_handler`/`policy_handler` sono **identici e condivisi** tra la versione terminale e quella web: `tello_pkg_web` li lancia direttamente dal package `tello_pkg`, senza copie.

> I vecchi nodi monolitici "tutto in uno" (`position_controller_VICON_VERSION`, `position_controller_web`) sono stati **rimossi**: la pipeline modulare qui sopra è l'unica versione mantenuta.

---

## 🧪 1. Diagnostica hardware — `tello_test`

Da eseguire **prima** di far volare il drone con la pipeline vera, per verificare la bontà dei dati Vicon e della telemetria Tello.

### `read_sensors` — solo monitoraggio, NESSUN comando di volo
Si connette a djitellopy solo per leggere batteria/quota, non invia mai `takeoff()`/`send_rc_control()`.
```bash
ros2 run tello_test read_sensors --ros-args \
    -p vicon_pose_topic:=/vicon/tello_42_boosted/tello_42_boosted
```
| Parametro | Default | Descrizione |
| :--- | :--- | :--- |
| `vicon_pose_topic` | `/vicon/tello_42_boosted/tello_42_boosted` | Topic `geometry_msgs/PoseStamped` del Tello dal Vicon. |
| `output_dir` | cartella dello script | Dove salvare CSV/PNG alla chiusura. |
| `save_csv` | `true` | Salva i log in CSV. |
| `save_plot` | `true` | Salva un PNG riassuntivo con i grafici della telemetria. |

### `takeoff_land` — test di volo isolato (SENZA policy)
Sequenza automatica takeoff → test movimento sui 3 assi → hover → land. Comandi manuali da terminale: `l`/`land` (atterra e chiudi), `q`/`quit` (chiudi, atterra se in volo).
```bash
ros2 run tello_test takeoff_land --ros-args \
    -p vicon_pose_topic:=/vicon/tello_42_boosted/tello_42_boosted
```
> `read_sensors` e `takeoff_land` usano entrambi djitellopy (porte UDP fisse 8889/8890): non possono girare insieme tra loro né insieme a un nodo di attuazione della pipeline (`vel_command_handler`/`vel_command_handler_web`).

---

## 🎮 2. Pipeline da terminale — `tello_pkg`

### Avvio: due launch file
| Launch file | Nodi lanciati | Uso |
| :--- | :--- | :--- |
| `no_motors.launch.py` | target_handler, observation_handler, policy_handler | Testare generazione target/osservazioni/policy **senza toccare il drone** (nessun motore, nessun armo). |
| `full_pipeline.launch.py` | + `vel_command_handler` | Pipeline completa, drone reale. |

```bash
ros2 launch tello_pkg full_pipeline.launch.py \
    target_mode:=variabile advance_mode:=manual num_queues:=-1 dof_mask_mode:=full
```

`vel_command_handler` si **connette** al drone appena parte (batteria/quota leggibili subito), ma **non decolla da solo**: aspetta un comando esplicito di `mission_console`.

### Controllo interattivo: `mission_console` (SEMPRE in un secondo terminale)
`ros2 launch` non inoltra lo stdin ai processi figli (limite noto di ROS2): il controllo interattivo va sempre lanciato a parte, nello stesso container:
```bash
ros2 run tello_pkg mission_console
```
All'avvio parte un **wizard sequenziale**: `target_mode`, `advance_mode`, `dof_mask_mode` (+ coordinate se `target_mode=custom`), `num_queues` — ogni risposta viene applicata subito (INVIO vuoto = non modificare). Poi resta nel loop operativo:

| Comando | Effetto |
| :--- | :--- |
| `INVIO` (riga vuota) | Avanza al prossimo waypoint (`/target_handler/advance`). |
| `l` / `land` | Atterraggio pulito (`/tello/land_request`). |
| `start` / `s` | Decolla / ripete il segnale di decollo (`/tello/start_request`). |
| `Ctrl+C` (nel terminale del **launch**) | Propaga SIGINT a tutti i nodi: `vel_command_handler` atterra sempre, anche se non è mai decollato. |

### Parametri (modificabili a runtime via `mission_console` o `ros2 param set`)
| Parametro | Nodo | Valori | Default | Descrizione |
| :--- | :--- | :--- | :--- | :--- |
| `target_mode` | target_handler | `singolo` \| `variabile` \| `custom` \| `hover` \| `aruco_target` | `variabile` | Vedi tabella modalità sotto. |
| `advance_mode` | target_handler | `manual` \| `auto` | `manual` | `manual`: avanza solo su comando esplicito. `auto`: avanza da solo quando dist/yaw sono sotto soglia per `TARGET_HOLD_TIME_S=1.2s` consecutivi. |
| `num_queues` | target_handler | intero | `-1` | Numero di code di 4 waypoint da completare prima di atterrare. `<=0` = infinito. |
| `custom_target_x/y/z/yaw` | target_handler | float (m / rad) | `0,0,1,0` | Target fisso, solo con `target_mode=custom`. Validati contro i limiti stanza (`ROOM_MIN`/`ROOM_MAX`), rifiutati se fuori. |
| `dof_mask_mode` | observation_handler | `full` \| `uniciclo` | `full` | Feature di osservazione (mai un mask sull'azione): `uniciclo` segnala alla policy di non usare $v_y$. |

### Modalità `target_mode`
| Modalità | Comportamento |
| :--- | :--- |
| `variabile` | 4 waypoint random distinti in sequenza dentro la stanza. |
| `singolo` | 1 target random, ripetuto su tutti gli slot della coda. |
| `custom` | 1 target fisso, dalle coordinate `custom_target_*` (validate contro i limiti stanza). |
| `hover` | Il target diventa la posizione/yaw esatti del drone al momento del takeoff. |
| `aruco_target` | Il target insegue in tempo reale il marker ArUco (subject Vicon `/vicon/aruco42/aruco42`), yaw fissato a 0. |

---

## 🌐 3. Pipeline con interfaccia web — `tello_pkg_web`

Stessa pipeline, un solo comando (nessun secondo terminale: il server web non ha il problema dello stdin non inoltrato):
```bash
ros2 launch tello_pkg_web web_pipeline.launch.py \
    target_mode:=variabile advance_mode:=manual num_queues:=-1 dof_mask_mode:=full
```
Poi apri il browser su:
```
http://<host>:8080/
```

`vel_command_handler_web` fonde tre ruoli nello stesso processo: attuazione/telemetria djitellopy (identica a `vel_command_handler`), il ruolo di `mission_console` esposto via HTTP invece che da wizard testuale, e lo stato per la dashboard 3D.

### Dashboard
- **Scena 3D** (Three.js): pallino blu = drone, pallino rosso = target attivo, cubetto viola = marker ArUco (sempre visibile e in movimento, indipendentemente dalla modalità attiva).
- **`target_mode=custom`**: compare uno slider per la quota Z e si può **cliccare direttamente sul pavimento della griglia 3D** per piazzare il target — validato contro i limiti stanza lato client e lato server.
- **Grafici**: velocità lineare/angolare body-frame, roll/pitch/yaw, in tempo reale.
- **Bottoni**: *Start algorithm* (unico modo per far decollare il drone — se non lo premi, il drone non si muove), *Land* (atterraggio incondizionato), *Next waypoint* (avanzamento manuale).

### Endpoint HTTP / WebSocket
| Endpoint | Metodo | Effetto |
| :--- | :--- | :--- |
| `/api/status` | GET | Ultimo snapshot di stato. |
| `/api/params` | POST | Imposta `dof_mask_mode`/`target_mode`/`advance_mode`/`num_queues` (solo a sessione idle). |
| `/api/custom_target` | POST `{x,y,z}` | Imposta il target custom, validato contro i limiti stanza, applicabile in qualunque momento. |
| `/api/start` | POST | Decolla (cancello di partenza esplicito). |
| `/api/land` | POST | Atterraggio d'emergenza incondizionato. |
| `/api/advance` | POST | Avanza al prossimo waypoint. |
| `/ws/state` | WebSocket | Broadcast continuo dello stato a **~10 Hz**. |

### Test del solo frontend, senza ROS/drone
```bash
python3 ros_ws/src/tello_pkg_web/tello_pkg_web/test_interface.py
```
Server finto (stesso formato JSON) che genera dati plausibili — utile per validare a occhio l'interfaccia senza hardware.

---

## 🔌 4. Topic e servizi ROS2 principali

| Topic | Tipo | Publisher | Subscriber | Frequenza |
| :--- | :--- | :--- | :--- | :---: |
| `/vicon/tello_42_boosted/tello_42_boosted` | `PoseStamped` | *bridge Vicon esterno* | target_handler, observation_handler, vel_command_handler_web, read_sensors, takeoff_land | rate del Vicon |
| `/vicon/aruco42/aruco42` | `PoseStamped` | *bridge Vicon esterno* | target_handler (se `aruco_target`), vel_command_handler_web (dashboard, sempre) | rate del Vicon |
| `targets` | `Float32MultiArray` (17 el.) | target_handler | observation_handler, vel_command_handler_web | 25 Hz |
| `observations` | `Float32MultiArray` (52 el.) | observation_handler | policy_handler | 25 Hz |
| `/tello/policy_action` | `Twist` | policy_handler | observation_handler (prev_action), vel_command_handler(_web) (attuazione) | 25 Hz |
| `/tello/flight_state` | `Bool` | vel_command_handler(_web) | target_handler | ad ogni transizione |
| `/tello/land_request` | `Empty` | target_handler, observation_handler | vel_command_handler(_web) | on‑demand |
| `/tello/start_request` | `Empty` | mission_console | vel_command_handler(_web) | on‑demand |
| `/target_handler/advance` | `Empty` | mission_console, vel_command_handler_web | target_handler | on‑demand |

| Servizio | Server | Client | Parametri |
| :--- | :--- | :--- | :--- |
| `/target_handler/set_parameters` | target_handler | mission_console, vel_command_handler_web | target_mode, advance_mode, num_queues, custom_target_x/y/z/yaw |
| `/observation_handler/set_parameters` | observation_handler | mission_console, vel_command_handler_web | dof_mask_mode |

---

## ⏱️ 5. Frequenze dei calcoli e loop di controllo

| Componente | Frequenza | Periodo | Descrizione |
| :--- | :---: | :---: | :--- |
| 🧠 target_handler / observation_handler / policy_handler — `control_loop` | **25 Hz** | 40 ms (`STEP_DT`) | Genera/avanza target → calcola le 52 osservazioni → inferenza rete → pubblica l'azione. |
| ⚙️ vel_command_handler(_web) — `watchdog_timer` | **5 Hz** | 200 ms | Se l'ultima `policy_action` è più vecchia di questo, forza hover (stick a zero). |
| 🔋 vel_command_handler(_web) — `telemetry_timer` | **1 Hz** | 1.0 s | Poll batteria/quota djitellopy, failsafe batteria critica (<15%). |
| 🖥️ vel_command_handler — `status_timer` | **1 Hz** | 1.0 s | Log di stato a terminale. |
| 🌐 vel_command_handler_web — `state_timer` | **10 Hz** | 100 ms | Snapshot broadcastato via WebSocket `/ws/state`. |
| 🛰️ Streaming posa Vicon | rate del Vicon | — | Event‑driven: ogni messaggio ricevuto aggiorna posizione/velocità/gravità proiettata (filtro passa-basso $\alpha=0.3$). |
| 📶 read_sensors / takeoff_land — telemetria | **10 Hz** | 100 ms | Poll dello state djitellopy. |

### Soglie di sicurezza
| Costante | Valore | Effetto |
| :--- | :--- | :--- |
| `POSE_TIMEOUT_S` | 0.5 s | Oltre questo, observation_handler smette di pubblicare osservazioni fresche. |
| `POSE_LOST_LAND_TIMEOUT_S` | 3.0 s | Vicon assente continuativamente oltre questo → atterraggio automatico. |
| `TELLO_LOST_LAND_TIMEOUT_S` | 3.0 s | Telemetria drone ferma oltre questo → atterraggio automatico. |
| `OBS_TIMEOUT_S` | 0.2 s | Oltre questo, policy_handler smette di pubblicare azioni. |
| `BATTERY_FAILSAFE_PCT` | 15% | Batteria sotto questa soglia → atterraggio automatico. |
| `MAX_LIN_VEL_MPS` / `MAX_YAW_RATE_RADPS` | 0.8 m/s / 1.0 rad/s | Cap di sicurezza sui comandi fisici inviati al drone. |
| `ROOM_MIN` / `ROOM_MAX` | [-2, -1.5, 0.1] / [2, 1.5, 2.0] m | Limiti della stanza: target random e custom sono sempre confinati qui dentro. |

> `Ctrl+C` atterra **sempre** il drone, in qualunque nodo di attuazione (`vel_command_handler`/`vel_command_handler_web`), anche se non è mai decollato — gestito con un handler SIGINT dedicato, non affidato al comportamento di default di `rclpy.spin()`.
