#include "HX711.h"

HX711 scale0;
HX711 scale1;
HX711 scale2;
HX711 scale3;

#define ENA 11
#define IN1 10
#define IN2 9

HX711 scales[4] = { scale0, scale1, scale2, scale3 };

const uint8_t dataPin[4] = { 4, 6, 8, 13 };
const uint8_t clockPin[4] = { 2, 5, 7, 12 };

float calib[4] = { 1.0, 1.0, 1.0, 1.0 };

float sum = 0;
float avg_running = 0;
float calibration_offset[] = {.0, .0, .0, .0};
float calibration_factor[] = {1.0/114, 1.0/114, 1.0/114, 1.0/114};

// ── Motor state machine (non-blocking) ──
bool motor_running = false;
unsigned long motor_start_ms = 0;
const unsigned long SHAKE_DURATION = 5000;  // ms (must match UI SHAKE_DURATION_S)

float sample(HX711 scale, int num_samples) {
  float sum_raw = 0;
  for (int i = 0; i < num_samples; i++) {
    sum_raw += scale.read();
  }
  return sum_raw / (float)num_samples;
}

void stop_motor() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 0);
  motor_running = false;
}


void setup()
{
  Serial.begin(115200);

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  // Motor brake immediately (prevents floating-pin spin during boot)
  stop_motor();

  for (int i = 0; i < 4; i++)
  {
    scales[i].begin(dataPin[i], clockPin[i]);
    scales[i].set_scale(calib[i]);
    scales[i].tare();
  }

  for (int i = 0; i < 4; i++){
    calibration_offset[i] = sample(scales[i], 10);
  }
}


void loop()
{
  // ── Motor timeout check: auto-stop after shake duration ──
  if (motor_running && (millis() - motor_start_ms >= SHAKE_DURATION)) {
    stop_motor();
  }

  // ── Serial commands ──
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "RESET") {
      stop_motor();  // safety: stop motor during reset
      for (int i = 0; i < 4; i++) {
        scales[i].tare();
        calibration_offset[i] = sample(scales[i], 10);
      }
      avg_running = 0;
      Serial.println("OK:RESET");
      return;
    }

    if (cmd == "SHAKE") {
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      analogWrite(ENA, 255);
      motor_running = true;
      motor_start_ms = millis();
      Serial.println("OK:SHAKE");
      return;
    }

    if (cmd == "STOP") {
      stop_motor();
      Serial.println("OK:STOP");
      return;
    }

    if (cmd == "WEIGHT") {
      Serial.print("WEIGHT:");
      Serial.println(avg_running);
      return;
    }
  }

  // ── Normal weight reading ──
  sum = 0;
  for (int i = 0; i < 4; i++){
    sum += (sample(scales[i], 1) - calibration_offset[i]) * calibration_factor[i];
  }
  sum /= 4;

  float diff = abs(sum - avg_running);
  float alpha = (diff > 5.0) ? 0.5 : 0.1;
  avg_running = (alpha * sum) + ((1.0 - alpha) * avg_running);

  Serial.print("Filtered:");
  Serial.println(avg_running);
}
