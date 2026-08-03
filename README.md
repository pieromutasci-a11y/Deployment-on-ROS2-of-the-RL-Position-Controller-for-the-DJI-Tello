# Tello ROS2 Node — Controller & Sensor Diagnostics

Questo repository contiene il package ROS2 `tello_node` per la telemetria, la diagnostica dei sensori e il controllo di posizione ad alto livello del drone **DJI Tello** utilizzando dati dal sistema **Vicon Mocap** e policy di Reinforcement Learning addestrate in Isaac Lab.

---

## 🚀 Guida Rapida all'Avvio (Docker)

### 1. Avviare il Container Docker
Dalla cartella principale del progetto sull'host:

```bash
./run.sh
```

Per aprire una seconda shell nel container attivo:
```bash
./exec.sh
```

### 2. Compilazione del Workspace ROS2
All'interno della shell del container (`/ros_workspace`):

```bash
colcon build --symlink-install
source install/setup.bash
```

---

## 🧪 1. Diagnostica Sensori e Plotting (`read_sensors`)

Prima di far volare il drone, è fortemente raccomandato eseguire il nodo di lettura dei sensori per verificare la bontà dei dati Vicon e della telemetria Tello. **Questo nodo NON invia comandi di volo al drone.**

```bash
ros2 run tello_node read_sensors --ros-args \
    -p vicon_pose_topic:=/vicon/tello/pose
```

### Parametri opzionali per `read_sensors`:
- `vicon_pose_topic` (default: `/vicon/tello/pose`): Topic su cui viene pubblicata la `PoseStamped` Vicon.
- `output_dir` (default: cartella dello script `ros_ws/src/tello_node/tello_node/`): Cartella di destinazione per i file salvati.
- `save_csv` (default: `true`): Salva i log dei dati in formato CSV alla chiusura del nodo.
- `save_plot` (default: `true`): Salva una figura PNG con 6 grafici riassuntivi della telemetria alla chiusura.

---

## 🎮 2. Controllore di Posizione Vicon (`position_controller_VICON_VERSION`)

### 📌 Comando Principale da Terminale

Ecco il comando completo per lanciare il nodo controllore di posizione con tutti i parametri configurabili:

```bash
ros2 run tello_node position_controller_VICON_VERSION --ros-args \
    -p dof_mask_mode:=full \
    -p target_mode:=variabile \
    -p advance_mode:=manual \
    -p num_queues:=-1 \
    -p vicon_pose_topic:=/vicon/tello/pose \
    -p cmd_vel_topic:=/tello/cmd_vel
```

---

### 📋 Dettaglio dei Parametri ROS2 (`--ros-args -p nome:=valore`)

| Parametro | Valori Ammessi | Default | Descrizione |
| :--- | :--- | :--- | :--- |
| `dof_mask_mode` | `full` \| `uniciclo` | `full` | **Maschera sui gradi di libertà** dei comandi $[v_x, v_y, v_z, \omega_z]$.<br>• `full`: applica $(1,1,1,1)$ — tutti i comandi attivi.<br>• `uniciclo`: applica $(1,0,1,1)$ — azzera hard il comando $v_y$ (no strafe). |
| `target_mode` | `variabile` \| `singolo` | `variabile` | **Modalità di generazione target**.<br>• `variabile`: genera 4 waypoint random distinti in sequenza.<br>• `singolo`: genera 1 solo target random ripetuto per tutti e 4 i waypoint della coda. |
| `advance_mode` | `manual` \| `auto` | `manual` | **Avanzamento tra i waypoint**.<br>• `manual`: l'utente deve premere **INVIO** da terminale per passare al prossimo waypoint.<br>• `auto`: avanzamento automatico quando il drone soddisfa la tolleranza di posizione e yaw per un tempo consecutivo `target_hold_time_s`. |
| `num_queues` | `intero` | `-1` | **Numero di code (cicli da 4 waypoint)** da completare prima di atterrare ed uscire automaticamente.<br>• `<= 0` (o `-1`): volo indefinito finché non viene inviato `q`.<br>• `N > 0`: ad esempio `3` esegue 3 code ($3 \times 4 = 12$ waypoint) e poi atterra. |
| `vicon_pose_topic` | `string` | `/vicon/tello/pose` | Nome del topic ROS2 `geometry_msgs/PoseStamped` dal sistema Vicon Tracker. |
| `cmd_vel_topic` | `string` | `/tello/cmd_vel` | Nome del topic ROS2 `geometry_msgs/Twist` dove pubblicare i comandi di velocità. |

---

### 💡 Esempi di Utilizzo Pratici

#### 🔹 Esempio 1: Modalità Manuale e Sicura (Consigliata per Primi Test di Volo)
Avanzamento manuale ad ogni waypoint tramite tasto `INVIO`, con tutti i gradi di libertà attivi e volo indefinito:

```bash
ros2 run tello_node position_controller_VICON_VERSION --ros-args \
    -p dof_mask_mode:=full \
    -p target_mode:=variabile \
    -p advance_mode:=manual \
    -p num_queues:=-1
```

#### 🔹 Esempio 2: Modalità Uniciclo (No Strafe / $v_y = 0$)
Disabilita il movimento laterale ($v_y$), utile per test con cinematica uniciclo:

```bash
ros2 run tello_node position_controller_VICON_VERSION --ros-args \
    -p dof_mask_mode:=uniciclo \
    -p target_mode:=variabile \
    -p advance_mode:=manual
```

#### 🔹 Esempio 3: Modalità Automatica con Limite di 2 Code (8 Waypoint totali)
Avanzamento automatico al raggiungimento dei target e atterraggio autonomo dopo 2 code completate:

```bash
ros2 run tello_node position_controller_VICON_VERSION --ros-args \
    -p dof_mask_mode:=full \
    -p target_mode:=variabile \
    -p advance_mode:=auto \
    -p num_queues:=2
```

---

### ⌨️ Comandi da Terminale Durante l'Esecuzione

Nel terminale dove è attivo il nodo controllore:
- **`INVIO`** *(riga vuota)*: Avanza al prossimo waypoint (valido solo in `advance_mode:=manual`).
- **`q`** oppure **`quit`** oppure **`exit`**: Invia l'atterraggio di emergenza immediato (`land`) e chiude il nodo in modo sicuro.

---

## ⏱️ 3. Frequenze dei Calcoli e Loop di Controllo

| Componente / Calcolo | Frequenza | Periodo ($\Delta t$) | Origine e Descrizione |
| :--- | :---: | :---: | :--- |
| 🧠 **Loop di Controllo RL (Policy)** | **25 Hz** | **40 ms** | `STEP_DT = 0.04`s. Frequenza di inferenza della rete neurale PyTorch e pubblicazione del comando `geometry_msgs/Twist` su `/tello/cmd_vel`. |
| 🎯 **Streaming Posa Vicon** | **100 – 200 Hz** | **5 – 10 ms** | Callback ROS2 `pose_cb`. Calcolo derivata velocità, gravità proiettata $\mathbf{g}_b$, e filtro passa-basso ($\alpha=0.3$) su $\mathbf{v}_b$. |
| 📡 **Tellopy IMU & MVO Log Data** | **10 – 20 Hz** | **50 – 100 ms** | Evento `EVENT_LOG_DATA` del Tello. Lettura giroscopio ($\text{gyro}_x, \text{gyro}_y, \text{gyro}_z$) e velocimetro ottico MVO. |
| 🔋 **Tellopy Flight Data** | **2 – 5 Hz** | **200 – 500 ms** | Evento `EVENT_FLIGHT_DATA` del Tello. Lettura percentuale batteria e quota sensore. |
| 🖥️ **Stampa Stato Terminale** | **1 Hz** *(0.5 Hz su `read_sensors`)* | **1.0 s** *(0.5 s)* | Timer ROS2 `status_cb`. Stampa a schermo i log di diagnostica, errore distanza/yaw e stato batteria. |
| ⌨️ **Polling Tastiera Terminale** | **5 Hz** | **200 ms** | Thread daemon con `select.select()` non bloccante per catturare la pressione di `INVIO` o `q`. |

