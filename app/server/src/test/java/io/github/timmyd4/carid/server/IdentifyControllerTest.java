package io.github.timmyd4.carid.server;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import io.github.timmyd4.carid.server.CarResult.Box;
import io.github.timmyd4.carid.server.CarResult.Prediction;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.List;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

/**
 * Tests the HTTP layer alone: request in, JSON out, without loading any models.
 *
 * <p>Two testing tools:
 * <ul>
 *   <li><b>Mockito</b> makes a fake CarPipeline that returns a canned answer. The real one
 *       needs 190 MB of models; here we only care that the controller passes the photo in
 *       and shapes the JSON correctly. (Like a stub implementation of an interface.)</li>
 *   <li><b>MockMvc</b> sends fake HTTP requests straight into Spring MVC, no real server or
 *       network, and lets us check the status code and the JSON fields.</li>
 * </ul>
 */
class IdentifyControllerTest {

    private CarPipeline pipeline;
    private MockMvc mvc;

    @BeforeEach
    void setUp() {
        pipeline = mock(CarPipeline.class);
        // standaloneSetup: just this one controller, without starting the whole application.
        mvc = MockMvcBuilders.standaloneSetup(new IdentifyController(pipeline)).build();
    }

    @Test
    void returnsVehiclesAsJson() throws Exception {
        LayeredAnswer answer = new LayeredAnswer(LayeredAnswer.Level.MODEL, "Toyota Corolla", 0.91f,
                "toyota", "corolla", null, null, null, null);
        CarResult car = new CarResult(new Box(10, 20, 110, 80), "car", 0.97f, true, answer,
                List.of(new Prediction("Toyota Corolla (2007-2009)", "toyota", "corolla", 2007, 2009, 0.82f)));
        // "When identify is called with any image and fallback=true, return this list."
        when(pipeline.identify(any(RgbImage.class), eq(true))).thenReturn(List.of(car));

        mvc.perform(multipart("/identify").file(photo(200, 100)))
                .andExpect(status().isOk())
                // jsonPath reads a field out of the JSON: $ is the root object.
                .andExpect(jsonPath("$.width").value(200))
                .andExpect(jsonPath("$.height").value(100))
                .andExpect(jsonPath("$.vehicles[0].vehicleType").value("car"))
                .andExpect(jsonPath("$.vehicles[0].box.left").value(10))
                // The enum goes out as its name; null fields stay in the JSON as null.
                .andExpect(jsonPath("$.vehicles[0].answer.level").value("MODEL"))
                .andExpect(jsonPath("$.vehicles[0].answer.text").value("Toyota Corolla"))
                .andExpect(jsonPath("$.vehicles[0].predictions[0].name").value("Toyota Corolla (2007-2009)"))
                .andExpect(jsonPath("$.vehicles[0].predictions[0].yearStart").value(2007));
    }

    @Test
    void liveModePassesFallbackFalse() throws Exception {
        when(pipeline.identify(any(RgbImage.class), eq(false))).thenReturn(List.of());

        mvc.perform(multipart("/identify").file(photo(64, 48)).param("fallback", "false"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.vehicles").isEmpty());
        // verify: check the controller really asked the pipeline for "no fallback".
        verify(pipeline).identify(any(RgbImage.class), eq(false));
    }

    @Test
    void rejectsFilesThatAreNotImages() throws Exception {
        MockMultipartFile text = new MockMultipartFile("photo", "notes.txt", "text/plain", "hello".getBytes());
        mvc.perform(multipart("/identify").file(text))
                .andExpect(status().isBadRequest());
    }

    /** A small blank PNG, uploaded under the form field name "photo". */
    private static MockMultipartFile photo(int width, int height) throws Exception {
        ByteArrayOutputStream png = new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR), "png", png);
        return new MockMultipartFile("photo", "test.png", "image/png", png.toByteArray());
    }
}
