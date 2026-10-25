package io.github.timmyd4.carid.client;

import com.google.gwt.core.client.EntryPoint;
import com.google.gwt.user.client.ui.Label;
import com.google.gwt.user.client.ui.RootPanel;

/**
 * Smoke-test entry point: the smallest GWT app that proves the toolchain works.
 *
 * <p>This Java never runs on a JVM in production. The GWT compiler translates it
 * into JavaScript, and the browser runs that. If `mvn package` turns this class
 * into app.nocache.js, then GWT 2.13.1 works on our JDK and the real UI can be
 * built on it. It gets replaced by the SmartGWT UI later.
 */
public class App implements EntryPoint {

    /**
     * Called by the browser once the page and the compiled script have loaded,
     * like main() for a desktop program. EntryPoint is the interface GWT looks for,
     * as named in module.gwt.xml.
     */
    @Override
    public void onModuleLoad() {
        // RootPanel.get() is the page's <body>. Adding a Label to it creates a
        // <div>Hello...</div> in the page, done in JavaScript at runtime.
        RootPanel.get().add(new Label("Hello from GWT! The Java-to-JavaScript compile works."));
    }
}
