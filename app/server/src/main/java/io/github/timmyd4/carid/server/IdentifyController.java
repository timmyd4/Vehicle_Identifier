package io.github.timmyd4.carid.server;

import ai.onnxruntime.OrtException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import javax.imageio.ImageIO;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

/**
 * The HTTP door into the pipeline: POST a photo, get every vehicle in it back as JSON.
 *
 * <pre>
 *   curl -F photo=@car.jpg http://localhost:8080/identify
 * </pre>
 *
 * <p>This is the first piece of code that makes the server do more than hand out files.
 * The phone UI (Phase 5) calls this same URL: the snapshot button sends one photo, and live
 * mode sends video frames several times a second.
 */
// @RestController: Spring routes matching HTTP requests to this class's methods, and turns
// whatever a method returns into JSON (records become {"field": value, ...}).
@RestController
public class IdentifyController {

    private final CarPipeline pipeline;

    // Spring injects the one CarPipeline it created at startup.
    public IdentifyController(CarPipeline pipeline) {
        this.pipeline = pipeline;
    }

    /**
     * The JSON body sent back.
     *
     * @param width     size of the photo as received, so the client can scale boxes to its screen
     * @param height    see width
     * @param millis    total server time, useful for watching live-mode speed
     * @param vehicles  every vehicle found, biggest first (empty in live mode if none)
     */
    public record IdentifyResponse(int width, int height, long millis, List<CarResult> vehicles) {
    }

    /**
     * @param photo    the image file. "multipart/form-data" is the standard way browsers and
     *                 curl upload files; Spring hands us the file as a MultipartFile.
     * @param fallback true (default, single photos): if no vehicle is found, classify the whole
     *                 photo anyway. Live mode sends false so an empty street returns nothing.
     */
    @PostMapping(path = "/identify", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public IdentifyResponse identify(@RequestParam("photo") MultipartFile photo,
                                     @RequestParam(defaultValue = "true") boolean fallback)
            throws IOException, OrtException {
        long start = System.nanoTime();

        BufferedImage image;
        try (InputStream in = photo.getInputStream()) {
            // ImageIO.read returns null (not an exception) for data it can't decode.
            image = ImageIO.read(in);
        }
        if (image == null) {
            // ResponseStatusException makes Spring answer with that HTTP status and message,
            // here 400 Bad Request: the client sent something that isn't a JPEG/PNG.
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Not an image (send a JPEG or PNG)");
        }

        List<CarResult> vehicles = pipeline.identify(RgbImage.from(image), fallback);
        return new IdentifyResponse(image.getWidth(), image.getHeight(),
                (System.nanoTime() - start) / 1_000_000, vehicles);
    }
}
