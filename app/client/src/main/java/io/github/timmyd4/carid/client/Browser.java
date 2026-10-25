package io.github.timmyd4.carid.client;

import jsinterop.annotations.JsFunction;
import jsinterop.annotations.JsPackage;
import jsinterop.annotations.JsType;

/**
 * Java "views" of the few browser JavaScript objects the UI needs.
 *
 * <p>GWT compiles Java to JavaScript, but some things only exist in JavaScript: the file
 * picker, uploading a file, parsing JSON. JsInterop lets us describe those JavaScript
 * objects as Java classes, then use them like normal Java. Nothing here gets compiled
 * into new code; each class is just a typed name for something the browser already has.
 *
 * <p>How to read the annotations:
 * <ul>
 *   <li>{@code @JsType(isNative = true)}: "this class already exists in JavaScript, don't
 *       generate it". Like declaring an interface whose implementation is provided elsewhere.</li>
 *   <li>{@code namespace = JsPackage.GLOBAL, name = "FormData"}: the JavaScript name is the
 *       global {@code FormData}, so {@code new FormData()} in Java becomes exactly
 *       {@code new FormData()} in JavaScript.</li>
 *   <li>Fields are JavaScript properties ({@code xhr.status}), and {@code native} methods
 *       are JavaScript methods ({@code xhr.send(form)}).</li>
 * </ul>
 */
final class Browser {

    private Browser() {
    }

    /** A JavaScript function with no arguments, e.g. an "onload" handler. A lambda fits it. */
    @JsFunction
    interface Callback {
        void call();
    }

    /** An {@code <input type="file">} element: opens the file picker, holds the chosen files. */
    @JsType(isNative = true, namespace = JsPackage.GLOBAL, name = "HTMLInputElement")
    static class FileInput {
        public String accept;        // which files the picker offers, e.g. "image/*"
        public String value;         // set to "" to allow picking the same file twice
        public FileList files;       // the chosen files
        public Callback onchange;    // runs after the user picks a file

        public native void click();  // opens the file picker
    }

    /** The list of chosen files. (A browser FileList, not a Java List.) */
    @JsType(isNative = true, namespace = JsPackage.GLOBAL)
    static class FileList {
        public int length;

        public native File item(int index);
    }

    /** One chosen file. The browser keeps its bytes; we just pass the File object around. */
    @JsType(isNative = true, namespace = JsPackage.GLOBAL)
    static class File {
        public String name;
    }

    /** The body of a file upload: field name -> value. Same format as an HTML form or curl -F. */
    @JsType(isNative = true, namespace = JsPackage.GLOBAL)
    static class FormData {
        public FormData() {
        }

        public native void append(String name, Object value);
    }

    /** Sends an HTTP request without reloading the page. */
    @JsType(isNative = true, namespace = JsPackage.GLOBAL)
    static class XMLHttpRequest {
        public int status;            // HTTP status code: 200 = OK, 400 = bad request, ...
        public String responseText;   // the response body, here our JSON
        public Callback onload;       // runs when the response arrives
        public Callback onerror;      // runs if the server can't be reached

        public XMLHttpRequest() {
        }

        public native void open(String method, String url);

        public native void send(Object body);
    }

    /** Makes a temporary "blob:" URL for a local file, so an {@code <img>} can show it. */
    @JsType(isNative = true, namespace = JsPackage.GLOBAL)
    static class URL {
        public static native String createObjectURL(Object file);

        public static native void revokeObjectURL(String url);
    }

    /** The browser's built-in JSON parser. */
    @JsType(isNative = true, namespace = JsPackage.GLOBAL)
    static class JSON {
        public static native Object parse(String text);
    }
}
