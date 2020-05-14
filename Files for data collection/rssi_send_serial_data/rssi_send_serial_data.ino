#include "ESP8266WiFi.h"

const int displayEnc=1;// set to 1 to display Encryption or 0 not to display

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(2000);
}

String data = "";

void loop() {

  int n = WiFi.scanNetworks();
  data = "";

  for (int i = 0; i < n; ++i) {
    if(WiFi.SSID(i) == "Tenda_21D848"){
      data += 'T';
      data += WiFi.RSSI(i);
    }
    if(WiFi.SSID(i) == "aviatorpranoy"){
      data += 'A';
      data += WiFi.RSSI(i);
    }
  }
  Serial.println(data);
  WiFi.scanDelete();  
}
