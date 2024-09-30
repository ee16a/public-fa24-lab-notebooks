unsigned long firstSensor = 0;    // first analog sensor
unsigned long secondSensor = 0;
unsigned int numAvgs = 1000;
int handshake = 0;

void setup()
{
  // start serial port at 115200 bps:
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for Leonardo only
  }
}

void loop() {
  // if we get a valid byte, read analog ins:
  if (Serial.available() > 0) {
    handshake = Serial.read();
    if (handshake == 57)  // Ascii code for numerical 9
      Serial.flush();
    else if (handshake == 54) { // Ascii code for 6 so serial monitor works
      firstSensor = 0;
      secondSensor = 0;
      for (int count = 0; count < numAvgs; count++) {  
        firstSensor += analogRead(A0);
        secondSensor += analogRead(A1);
        delay(0.1);
      }
      Serial.println((firstSensor + secondSensor) / (2 * numAvgs), DEC);
    }
  }
}
