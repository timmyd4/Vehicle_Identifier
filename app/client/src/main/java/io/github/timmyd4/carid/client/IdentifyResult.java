package io.github.timmyd4.carid.client;

import jsinterop.annotations.JsPackage;
import jsinterop.annotations.JsType;

/**
 * The JSON that POST /identify sends back, as Java classes.
 *
 * <p>These mirror the server's records (IdentifyController.IdentifyResponse, CarResult) field
 * for field. {@code name = "Object"} says "this is a plain JavaScript object", so the result
 * of JSON.parse() can be cast straight to these types and read like Java fields. No
 * hand-written parsing code needed.
 */
@JsType(isNative = true, namespace = JsPackage.GLOBAL, name = "Object")
class IdentifyResult {
    public int width;
    public int height;
    public double millis;
    public Vehicle[] vehicles;

    @JsType(isNative = true, namespace = JsPackage.GLOBAL, name = "Object")
    static class Vehicle {
        public Box box;
        public String vehicleType;
        public double detectorScore;
        public boolean detected;
        public Prediction[] predictions;
    }

    @JsType(isNative = true, namespace = JsPackage.GLOBAL, name = "Object")
    static class Box {
        public int left;
        public int top;
        public int right;
        public int bottom;
    }

    @JsType(isNative = true, namespace = JsPackage.GLOBAL, name = "Object")
    static class Prediction {
        public String name;
        public double probability;
    }
}
