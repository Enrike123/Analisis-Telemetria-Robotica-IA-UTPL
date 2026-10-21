
PRAGMA foreign_keys = ON;

CREATE TABLE motores (
    motor_id INTEGER PRIMARY KEY,
    pata TEXT NOT NULL,
    articulacion TEXT NOT NULL
);

CREATE TABLE mediciones (
    medicion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    motor_id INTEGER NOT NULL,
    instante INTEGER NOT NULL,
    tiempo_ros REAL NOT NULL,
    tiempo_relativo_s REAL NOT NULL,
    posicion_rad REAL NOT NULL,
    velocidad_rad_s REAL NOT NULL,
    torque_Nm REAL NOT NULL,
    temperatura_C INTEGER NOT NULL,
    sesion TEXT NOT NULL,

    FOREIGN KEY (motor_id)
        REFERENCES motores(motor_id),

    UNIQUE (sesion, instante, motor_id)
);
