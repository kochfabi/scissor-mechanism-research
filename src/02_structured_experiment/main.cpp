#include <Arduino.h>
#include <HX711.h>

HX711 sensor;

const uint8_t DATA_PIN  = 3;
const uint8_t CLOCK_PIN = 2;
const float   GRAVITY   = 9.7976f; // m/s², Tokyo

void waitForEnter();
void calibrate();

void setup() {
    Serial.begin(9600);
    sensor.begin(DATA_PIN, CLOCK_PIN);
    calibrate();
    Serial.println("READY");
}

void loop() {
    float force_N  = sensor.get_units(5);
    float weight_g = force_N * 1000.0f / GRAVITY;

    Serial.print(millis() / 1000.0f, 3);
    Serial.print(",");
    Serial.print(force_N, 6);
    Serial.print(",");
    Serial.println(weight_g, 4);
}

void calibrate() {
    // Step 1: wait for Python to confirm user removed all weight
    waitForEnter();
    sensor.tare(100);
    Serial.print("OFFSET:");
    Serial.println(sensor.get_offset());

    // Step 2: wait for Python to send known weight in grams
    uint32_t weight_g = 0;
    while (true) {
        if (Serial.available()) {
            char c = Serial.read();
            if (c == '\n') break;
            if (isdigit(c)) weight_g = weight_g * 10 + (c - '0');
        }
    }
    float force_N = weight_g / 1000.0f * GRAVITY;
    sensor.calibrate_scale(force_N, 100);
    Serial.print("SCALE:");
    Serial.println(sensor.get_scale(), 6);
}

void waitForEnter() {
    while (true) {
        if (Serial.available() && Serial.read() == '\n') break;
    }
}