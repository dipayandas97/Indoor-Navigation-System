#include "ESP8266WiFi.h"

const int displayEnc=1;// set to 1 to display Encryption or 0 not to display

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(2000);
}

void loop() {

  int n = WiFi.scanNetworks();
  if (n == 0) {
    Serial.println("!");
  } 
  else {
    for (int i = 0; i < n; ++i) {
      Serial.print(WiFi.SSID(i));
      Serial.print(':');
      Serial.print(WiFi.RSSI(i));
      Serial.print(' ');
    }
  }
  Serial.println("");
  WiFi.scanDelete();  
}
