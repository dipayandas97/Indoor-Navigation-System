#include "ESP8266WiFi.h"

String ssid = "Tenda_21D848";
String password = "08042020";

void setup() {
  
  Serial.begin(115200);
  //WiFi.mode(WIFI_STA);
  delay(10);
  Serial.println("Connecting...");
  WiFi.begin(ssid,password);
  
}

void loop() {
  
 Serial.println(WiFi.RSSI());
 delay(10);

}
