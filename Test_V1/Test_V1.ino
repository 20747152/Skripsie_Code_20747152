#include "HX711.h"

#define Dout_Upper 2
#define CLK_Upper 3
#define Dout_Lower 4
#define CLK_Lower 5
#define Read_Button 6
#define Stop_Button 7
#define Read_LED 8
#define Ready_LED 9

HX711 scale_1;
HX711 scale_2;

float calibration_factor_Upper = 109; //-7050 worked for my 440lb max scale setup
float calibration_factor_Lower = 108.5;
int start_timer = millis();
int counter = 0;
int Clock_counter = millis();
void setup() {
  Serial.begin(9600);
  scale_1.begin(Dout_Upper, CLK_Upper);
  scale_2.begin(Dout_Lower, CLK_Lower);
  pinMode(6, INPUT);
  pinMode(7, INPUT);
  pinMode(8, OUTPUT);
  pinMode(9, OUTPUT);
  digitalWrite(9, HIGH);
  digitalWrite(8, LOW);
  scale_1.set_scale();
  scale_2.set_scale();
  scale_1.tare(); //Reset the scale to 0
  scale_2.tare();

  long zero_factor = scale_1.read_average(); //Get a baseline reading
  //Serial.print("Zero factor: "); //This can be used to remove the need to tare the scale. Useful in permanent scale projects.
  //Serial.println(zero_factor);
}

void loop() {

  scale_1.set_scale(calibration_factor_Upper); //Adjust to this calibration factor
  scale_2.set_scale(calibration_factor_Lower);
  //counter = millis() - start_timer;

  if (digitalRead(Read_LED) == HIGH) {
    //Serial.println("Gauge 1 : Gauge 2");
    Serial.print(scale_1.get_units(), 1)*1000;
    Serial.print(" : ");
    Serial.print(scale_2.get_units(), 1)*1000;
    Serial.println();
    //start_timer = millis();
  }
  counter = millis()-Clock_counter;
  
  if((digitalRead(Ready_LED) == HIGH) && (digitalRead(Stop_Button) == HIGH) && (counter >= 200) ){
    scale_1.tare();
    scale_2.tare();
  }
  
  
  
  if ((digitalRead(Ready_LED) == HIGH) && (digitalRead(Read_Button) == HIGH)) {
    digitalWrite(Read_LED, HIGH);
    digitalWrite(Ready_LED, LOW);
    
  }
  if ((digitalRead(Read_LED) == HIGH) && (digitalRead(Stop_Button) == HIGH)) {
    digitalWrite(Ready_LED, HIGH);
    digitalWrite(Read_LED, LOW);
    Clock_counter = millis();
  }



  //  if (counter >= 250) {
  //    Serial.println("Gauge 1 : Gauge 2");
  //    Serial.print(scale_1.get_units(), 1);
  //    Serial.print(" : ");
  //    Serial.print(scale_2.get_units(), 1);
  //    Serial.println();
  //    start_timer = millis();
  //
  //  }

  // Serial.print("Reading: ");
  // Serial.print(scale_1.get_units(), 1);
  // Serial.print(" kg"); //Change this to kg and re-adjust the calibration factor if you follow SI units like a sane person
  //Serial.print(" calibration_factor: ");
  //Serial.print(calibration_factor);
  //Serial.println();
}
