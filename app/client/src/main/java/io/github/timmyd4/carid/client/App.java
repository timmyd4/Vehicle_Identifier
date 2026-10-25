package io.github.timmyd4.carid.client;

import com.google.gwt.core.client.EntryPoint;
import com.google.gwt.dom.client.Document;
import com.google.gwt.safehtml.shared.SafeHtmlUtils;
import com.smartgwt.client.widgets.HTMLFlow;
import com.smartgwt.client.widgets.IButton;
import com.smartgwt.client.widgets.Label;
import com.smartgwt.client.widgets.Window;
import com.smartgwt.client.widgets.layout.HLayout;
import com.smartgwt.client.widgets.layout.VLayout;
import io.github.timmyd4.carid.client.Browser.File;
import io.github.timmyd4.carid.client.Browser.FileInput;
import io.github.timmyd4.carid.client.Browser.FormData;
import io.github.timmyd4.carid.client.Browser.JSON;
import io.github.timmyd4.carid.client.Browser.URL;
import io.github.timmyd4.carid.client.Browser.XMLHttpRequest;
import io.github.timmyd4.carid.client.IdentifyResult.Box;
import io.github.timmyd4.carid.client.IdentifyResult.Prediction;
import io.github.timmyd4.carid.client.IdentifyResult.Vehicle;

/**
 * A simple upload page: pick a car photo, see what the server thinks it is.
 *
 * <p>Flow: button -> file picker -> upload the photo to POST /identify -> a popup window
 * shows the photo with a colored box around each vehicle, and the guesses in matching colors.
 *
 * <p>This Java runs in the browser (compiled to JavaScript by GWT). The server does the
 * actual recognition; this page only sends the photo and draws the answer.
 */
public class App implements EntryPoint {

    /** How wide the photo is shown in the popup, in screen pixels. */
    private static final int SHOW_WIDTH = 480;

    /** One color per vehicle, so each box matches its list of guesses. */
    private static final String[] COLORS = {"#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa"};

    private final Label status = new Label();

    @Override
    public void onModuleLoad() {
        Label title = new Label("<h2>Car Identifier</h2>Upload a photo of a car to see what the model thinks it is.");
        title.setHeight(70);

        // A hidden <input type="file">: the browser's own file picker. We never show it;
        // the SmartGWT button below "clicks" it for us, so the page keeps SmartGWT's look.
        FileInput picker = (FileInput) (Object) Document.get().createFileInputElement();
        picker.accept = "image/*";
        picker.onchange = () -> {
            if (picker.files.length > 0) {
                upload(picker.files.item(0));
            }
            picker.value = "";   // so choosing the same file again still triggers onchange
        };

        IButton button = new IButton("Upload photo");
        button.setWidth(160);
        button.addClickHandler(event -> picker.click());

        status.setHeight(30);

        VLayout page = new VLayout(10);
        page.setPadding(20);
        page.setMembers(title, button, status);
        page.draw();
    }

    /** Send the photo to the server; show the result (or the error) when the answer arrives. */
    private void upload(File file) {
        status.setContents("Identifying " + SafeHtmlUtils.htmlEscape(file.name) + "...");

        // The same request as: curl -F photo=@file http://localhost:8080/identify
        FormData form = new FormData();
        form.append("photo", file);

        XMLHttpRequest request = new XMLHttpRequest();
        // A relative URL: "identify" on the same server that served this page.
        request.open("POST", "identify");
        // These callbacks run LATER, when the response arrives. The browser doesn't wait
        // (it would freeze the page), so we say what to do once it's done. That's
        // "asynchronous" code, like a listener in Swing.
        request.onload = () -> {
            if (request.status == 200) {
                status.setContents("");
                showResult(file, (IdentifyResult) JSON.parse(request.responseText));
            } else {
                status.setContents("Server error " + request.status);
            }
        };
        request.onerror = () -> status.setContents("Could not reach the server.");
        request.send(form);
    }

    /** A popup with the photo, a box per vehicle, and each vehicle's guesses. */
    private void showResult(File file, IdentifyResult result) {
        // Show the local file directly (no need to download it back from the server).
        String imageUrl = URL.createObjectURL(file);

        // The server's boxes are in the photo's real pixels; the photo is shown smaller,
        // so every coordinate gets multiplied by the same scale.
        double scale = Math.min(1.0, (double) SHOW_WIDTH / result.width);
        int shownWidth = (int) Math.round(result.width * scale);
        int shownHeight = (int) Math.round(result.height * scale);

        StringBuilder picture = new StringBuilder();
        // position:relative on the container lets the boxes use position:absolute inside it.
        picture.append("<div style='position:relative;width:").append(shownWidth)
                .append("px;height:").append(shownHeight).append("px'>");
        // image-orientation:none shows the pixels as stored, ignoring the phone's "rotate
        // me" flag, because that's how the server sees them. Otherwise boxes could be drawn
        // on a rotated photo in the wrong place.
        picture.append("<img src='").append(imageUrl).append("' style='image-orientation:none;width:")
                .append(shownWidth).append("px;height:").append(shownHeight).append("px'>");

        StringBuilder guesses = new StringBuilder();
        Vehicle[] vehicles = result.vehicles;
        for (int i = 0; i < vehicles.length; i++) {
            Vehicle v = vehicles[i];
            String color = COLORS[i % COLORS.length];
            if (v.detected) {
                Box b = v.box;
                picture.append("<div style='position:absolute;border:3px solid ").append(color)
                        .append(";left:").append(Math.round(b.left * scale))
                        .append("px;top:").append(Math.round(b.top * scale))
                        .append("px;width:").append(Math.round((b.right - b.left) * scale))
                        .append("px;height:").append(Math.round((b.bottom - b.top) * scale))
                        .append("px'></div>");
                guesses.append("<h3 style='color:").append(color).append("'>Vehicle ").append(i + 1)
                        .append(" (").append(v.vehicleType).append(", ").append(percent(v.detectorScore))
                        .append(" sure it's a vehicle)</h3>");
            } else {
                guesses.append("<h3>No vehicle found: guessing from the whole photo</h3>");
            }
            guesses.append("<ol>");
            for (Prediction p : v.predictions) {
                guesses.append("<li>").append(SafeHtmlUtils.htmlEscape(p.name))
                        .append(": <b>").append(percent(p.probability)).append("</b></li>");
            }
            guesses.append("</ol>");
        }
        picture.append("</div>");

        HTMLFlow photo = new HTMLFlow(picture.toString());
        photo.setWidth(shownWidth);
        HTMLFlow list = new HTMLFlow(guesses.toString());
        list.setWidth(320);

        HLayout content = new HLayout(15);
        content.setPadding(10);
        content.setMembers(photo, list);

        // A SmartGWT Window: a draggable popup with a title bar and close button.
        Window popup = new Window();
        popup.setTitle(SafeHtmlUtils.htmlEscape(file.name) + " (" + Math.round(result.millis) + " ms)");
        popup.setWidth(shownWidth + 370);
        popup.setHeight(Math.max(shownHeight, 300) + 60);
        popup.setIsModal(true);          // block the page behind it until closed
        popup.setShowModalMask(true);    // and dim it
        popup.addItem(content);
        popup.addCloseClickHandler(event -> {
            popup.destroy();
            URL.revokeObjectURL(imageUrl);   // free the temporary image URL
        });
        popup.centerInPage();
        popup.show();
    }

    private static String percent(double fraction) {
        return Math.round(fraction * 1000) / 10.0 + "%";
    }
}
