package io.github.timmyd4.carid.client;

import com.google.gwt.core.client.EntryPoint;
import com.smartgwt.client.util.SC;
import com.smartgwt.client.widgets.IButton;
import com.smartgwt.client.widgets.Label;
import com.smartgwt.client.widgets.layout.VLayout;

/**
 * Smoke-test entry point: proves SmartGWT compiles with our GWT 2.13.1 on JDK 25.
 *
 * <p>This Java never runs on a JVM in production. The GWT compiler translates it
 * into JavaScript, and the browser runs that. It gets replaced by the real UI
 * (phone camera view, history grid) later.
 *
 * <p>Note the imports: Label and the layout now come from com.smartgwt, not GWT's
 * own com.google.gwt.user.client.ui. SmartGWT has its own complete widget set.
 */
public class App implements EntryPoint {

    /**
     * Called by the browser once the page and the compiled script have loaded,
     * like main() for a desktop program.
     */
    @Override
    public void onModuleLoad() {
        Label label = new Label("Hello from SmartGWT!");
        label.setHeight(30);

        // IButton is SmartGWT's themed button (the "I" means it uses the theme's images).
        IButton button = new IButton("Click me");
        // A lambda works here because ClickHandler has a single method, the same as a
        // Swing ActionListener. GWT translates lambdas to JavaScript functions.
        // SC.say() shows a SmartGWT popup dialog, like JOptionPane.showMessageDialog.
        button.addClickHandler(event -> SC.say("SmartGWT works on GWT 2.13.1!"));

        // VLayout stacks its members vertically, like a Swing BoxLayout on the Y axis.
        VLayout layout = new VLayout(10);   // 10px gap between members
        layout.setPadding(20);
        layout.setMembers(label, button);

        // SmartGWT widgets draw themselves onto the page. That replaces GWT's RootPanel.get().add(...).
        layout.draw();
    }
}
