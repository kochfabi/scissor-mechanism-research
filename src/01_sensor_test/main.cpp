#include <Arduino.h>
#include <HX711.h>

HX711 sensor;

// Pins
const uint8_t DATA_PIN = 3;
const uint8_t CLOCK_PIN = 2;

// Parameters
float g = 9.7976; // m/s^2, standard gravity

// Function prototypes
void calibrate();
void waitforenter();

void setup(){
  Serial.begin(9600);
  sensor.begin(DATA_PIN, CLOCK_PIN);
  calibrate();
  Serial.println("timestamp_s,force_N,weight_g");
}

void loop() {
  // read force sensor
  float force_N = sensor.get_units(5);
  float weight_g = force_N * 1000.0 / g;
  
  // print measurement as CSV: timestamp_s,force_N,weight_g
  Serial.print(millis() / 1000.0, 3);
  Serial.print(",");
  Serial.print(force_N, 5);
  Serial.print(",");
  Serial.println(weight_g, 3);
}

void calibrate() {
  // Flush Serial input
  while (Serial.available()) Serial.read();
  waitforenter();

  Serial.println("======================CALIBRATION======================");
  Serial.println("Remove all weight from the loadcell and press enter\n");
  waitforenter();

  // Determine offset
  Serial.println("Determine zero weight offset");
  sensor.tare(100); //average 100 measurements
  int32_t offset = sensor.get_offset();
  Serial.print("OFFSET: ");
  Serial.println(offset);

  // Determine scale
  Serial.println("\nPlace known weight on loadcell");
  while (Serial.available()) Serial.read();
  Serial.println("Enter the weight [g] and press enter");
  uint32_t weight = 0;
  while (Serial.peek() != '\n')
  {
    if (Serial.available())
    {
      char ch = Serial.read();
      if (isdigit(ch))
      {
        weight *= 10;
        weight = weight + (ch - '0');
      }
    }
  }
  Serial.print("WEIGHT: ");
  Serial.println(weight);
  float force = weight / 1000.0 * g ; // convert grams to kilograms
  sensor.calibrate_scale(force, 100);
  float scale = sensor.get_scale();
  Serial.print("SCALE:  ");
  Serial.println(scale, 6);

  Serial.print("\nuse scale.set_offset(");
  Serial.print(offset);
  Serial.print("); and scale.set_scale(");
  Serial.print(scale, 6);
  Serial.print(");\n");
  Serial.println("in the setup of your project");
  delay(2000);
  Serial.println("\n");
  Serial.println("======================MEASUREMENT======================");
}

void waitforenter() {
  while (true) {
    if (Serial.available()) {
      char c = Serial.read();
      if (c == '\n') break;
    }
  }
}