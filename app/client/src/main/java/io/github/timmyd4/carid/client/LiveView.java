package io.github.timmyd4.carid.client;

import com.google.gwt.core.client.Scheduler;
import com.google.gwt.dom.client.Document;
import com.google.gwt.dom.client.Element;
import com.google.gwt.safehtml.shared.SafeHtmlUtils;
import io.github.timmyd4.carid.client.Browser.Canvas;
import io.github.timmyd4.carid.client.Browser.Clickable;
import io.github.timmyd4.carid.client.Browser.FormData;
import io.github.timmyd4.carid.client.Browser.Global;
import io.github.timmyd4.carid.client.Browser.JSON;
import io.github.timmyd4.carid.client.Browser.MediaStream;
import io.github.timmyd4.carid.client.Browser.MediaStreamTrack;
import io.github.timmyd4.carid.client.Browser.Video;
import io.github.timmyd4.carid.client.Browser.XMLHttpRequest;
import io.github.timmyd4.carid.client.IdentifyResult.Box;
import io.github.timmyd4.carid.client.IdentifyResult.Vehicle;

/**
 * Live mode: the camera fills the screen, and every car in view gets a box with its name.
 *
 * <p>THE LOOP, over and over until Close:
 * <ol>
 *   <li>grab the current video frame onto a canvas, shrunk to {@link #FRAME_WIDTH} pixels wide</li>
 *   <li>turn it into a JPEG and POST it to /identify, the same endpoint as the upload page</li>
 *   <li>when the answer arrives, draw the boxes over the video, then go back to step 1</li>
 * </ol>
 *
 * <p>WHY "send the next frame only when the answer is back" (one request at a time): the phone
 * films 30 frames a second, the server handles maybe 5. Sending all 30 would pile up a queue,
 * and the boxes would show where the cars were seconds ago. This way the speed adjusts itself:
 * a fast server or network gives more updates per second, a slow one fewer, and the boxes are
 * never more than one answer behind.
 *
 * <p>WHY shrink to 640 pixels: a phone frame can be 1920x1080. The detector and classifier
 * shrink everything to a few hundred pixels anyway, so sending more only costs upload time.
 *
 * <p>This is plain browser DOM (a full-screen div on top of the page), not SmartGWT widgets:
 * it needs a live {@code <video>}, which is easier to handle directly.
 */
final class LiveView {

    private static final int FRAME_WIDTH = 640;
    /** Boxes older than this are removed: on a slow connection they'd show where cars WERE. */
    private static final int STALE_MILLIS = 1500;
    private static final String[] COLORS = {"#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa"};

    private final Document document = Document.get();
    private final Element root = document.createDivElement();
    private final Element boxes = document.createDivElement();
    private final Element status = document.createDivElement();
    private final Video video = (Video) (Object) document.createElement("video");
    private final Canvas canvas = (Canvas) (Object) document.createElement("canvas");

    private MediaStream stream;
    private boolean running;
    private double lastAnswerTime;
    private double sendTime;      // when the current frame was POSTed, to time the round trip
    private int answerCount;      // goes up with every answer drawn (see clearIfStale)

    /** Open the live view over the page and start the camera. */
    static void open() {
        new LiveView().start();
    }

    private LiveView() {
        // Full screen, black, above everything SmartGWT draws (it uses big z-index values).
        root.setAttribute("style", "position:fixed;inset:0;background:#000;z-index:2000000;"
                + "display:flex;align-items:center;justify-content:center;font:15px system-ui,sans-serif");

        // The "stage" is exactly as big as the video on screen, so boxes placed inside it in
        // % or px line up with the picture. inline-block makes it shrink to fit the video.
        Element stage = document.createDivElement();
        stage.setAttribute("style", "position:relative;display:inline-block");
        Element videoElement = (Element) (Object) video;
        videoElement.setAttribute("style", "display:block;max-width:100vw;max-height:100vh");
        video.muted = true;          // browsers only autoplay video that's muted
        video.autoplay = true;
        video.playsInline = true;
        boxes.setAttribute("style", "position:absolute;inset:0;pointer-events:none");
        stage.appendChild(videoElement);
        stage.appendChild(boxes);

        status.setAttribute("style", "position:fixed;top:10px;left:10px;color:#fff;"
                + "background:rgba(0,0,0,.6);padding:6px 10px;border-radius:6px");
        Element close = document.createElement("button");
        close.setInnerText("Close");
        close.setAttribute("style", "position:fixed;top:10px;right:10px;font-size:16px;padding:8px 16px");
        ((Clickable) (Object) close).onclick = this::stop;

        root.appendChild(stage);
        root.appendChild(status);
        root.appendChild(close);
        document.getBody().appendChild(root);
    }

    private void start() {
        if (Global.navigator.mediaDevices == null) {
            // Browsers only give the camera to secure pages: https://..., or http://localhost.
            status.setInnerText("The camera needs a secure page: open this over https, or on this PC at localhost.");
            return;
        }
        status.setInnerText("Asking for the camera...");
        // facingMode "environment" = the BACK camera on a phone (the PC just uses its webcam).
        // "ideal" means "if you can": the browser picks the closest it has instead of failing.
        Object constraints = JSON.parse("{\"audio\":false,\"video\":{\"facingMode\":{\"ideal\":\"environment\"},"
                + "\"width\":{\"ideal\":1280},\"height\":{\"ideal\":720}}}");
        Global.navigator.mediaDevices.getUserMedia(constraints).then(cameraStream -> {
            stream = cameraStream;
            running = true;
            video.onloadedmetadata = this::sendFrame;   // first frame once the video size is known
            video.srcObject = cameraStream;
        }, error -> status.setInnerText("No camera: " + error + " (allowed in the browser's site settings?)"));
    }

    /** Steps 1 + 2 of the loop: grab the frame, encode it, send it. */
    private void sendFrame() {
        if (!running || video.videoWidth == 0) {
            return;
        }
        canvas.width = FRAME_WIDTH;
        canvas.height = (int) Math.round((double) video.videoHeight * FRAME_WIDTH / video.videoWidth);
        canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(this::post, "image/jpeg", 0.8);
    }

    private void post(Object jpeg) {
        FormData form = new FormData();
        form.append("photo", jpeg);
        XMLHttpRequest request = new XMLHttpRequest();
        // fallback=false: no car in view -> no answer, instead of guessing from the whole frame.
        request.open("POST", "identify?fallback=false");
        request.onload = () -> {
            if (!running) {
                return;   // closed while this answer was on its way
            }
            if (request.status == 200) {
                draw((IdentifyResult) JSON.parse(request.responseText));
                sendFrame();                    // step 3: straight on to the next frame
            } else {
                boxes.setInnerHTML("");         // don't leave old boxes up while we retry
                status.setInnerText("Server error " + request.status + ", retrying...");
                retrySoon();
            }
        };
        request.onerror = () -> {
            boxes.setInnerHTML("");
            status.setInnerText("Can't reach the server, retrying...");
            retrySoon();
        };
        sendTime = System.currentTimeMillis();
        request.send(form);
    }

    /** After an error, wait a second before the next try instead of hammering the server. */
    private void retrySoon() {
        Scheduler.get().scheduleFixedDelay(() -> {
            sendFrame();
            return false;   // false = run once, don't repeat
        }, 1000);
    }

    /** Step 3: a box and a name tag per vehicle, scaled from frame pixels to screen pixels. */
    private void draw(IdentifyResult result) {
        double scale = (double) video.clientWidth / result.width;
        StringBuilder html = new StringBuilder();
        for (int i = 0; i < result.vehicles.length; i++) {
            Vehicle v = result.vehicles[i];
            if (!v.detected) {
                continue;
            }
            Box b = v.box;
            String color = COLORS[i % COLORS.length];
            boolean notSure = "NOT_SURE".equals(v.answer.level);
            html.append("<div style='position:absolute;border:3px solid ").append(color)
                    .append(";left:").append(Math.round(b.left * scale))
                    .append("px;top:").append(Math.round(b.top * scale))
                    .append("px;width:").append(Math.round((b.right - b.left) * scale))
                    .append("px;height:").append(Math.round((b.bottom - b.top) * scale)).append("px'>")
                    // The name tag sits just above the box (bottom:100% = its bottom edge on the box's top).
                    .append("<div style='position:absolute;bottom:100%;left:-3px;white-space:nowrap;padding:2px 6px;")
                    .append("color:#fff;background:").append(notSure ? "#666" : color).append("'>")
                    .append(SafeHtmlUtils.htmlEscape(v.answer.text)).append(" ")
                    .append(Math.round(v.answer.confidence * 100)).append("%</div></div>");
        }
        boxes.setInnerHTML(html.toString());

        // Updates per second = 1 / time since the previous answer.
        double now = System.currentTimeMillis();
        String speed = lastAnswerTime > 0 ? Math.round(10000 / (now - lastAnswerTime)) / 10.0 + " updates/s, " : "";
        lastAnswerTime = now;
        // WHERE THE TIME GOES: the round trip (send -> answer) = the server's work + the network
        // (uploading the frame, downloading the answer). The server reports its part, so the
        // rest is the network. If "network" is the big number, a faster server won't help.
        double network = Math.max(0, now - sendTime - result.millis);
        status.setInnerText(speed + "server " + Math.round(result.millis) + " ms, network "
                + Math.round(network) + " ms" + (result.vehicles.length == 0 ? " | no car in view" : ""));

        // STALE BOXES: these boxes show where the cars were when the frame was taken. If
        // the next answer is slow (bad signal), the camera may already point somewhere
        // else, so take them down after STALE_MILLIS unless a newer answer replaced them.
        int thisAnswer = ++answerCount;
        Scheduler.get().scheduleFixedDelay(() -> {
            clearIfStale(thisAnswer);
            return false;   // run once
        }, STALE_MILLIS);
    }

    /** Remove the boxes if answer number {@code drawn} is still the newest one on screen. */
    private void clearIfStale(int drawn) {
        if (running && answerCount == drawn) {
            boxes.setInnerHTML("");
            status.setInnerText("Waiting for the server (slow connection)...");
        }
    }

    /** Close: stop the loop, turn the camera off, remove the view. */
    private void stop() {
        running = false;
        if (stream != null) {
            for (MediaStreamTrack track : stream.getTracks()) {
                track.stop();
            }
        }
        root.removeFromParent();
    }
}
