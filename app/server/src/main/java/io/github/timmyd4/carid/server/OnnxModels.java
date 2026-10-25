package io.github.timmyd4.carid.server;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import jakarta.annotation.PreDestroy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

/**
 * Loads the two ONNX models once, when the server starts, and keeps them in memory.
 *
 * <p>Big picture: the pipeline is detector (find every car in the frame) then classifier
 * (name each one). Both were trained in PyTorch and exported to ONNX by
 * training/export_onnx.py; ONNX Runtime runs them here in Java with no Python involved.
 *
 * <p>Why load at startup: creating a session parses ~100 MB and optimizes the model graph,
 * which takes about a second. Doing that per request would make live mode impossible, so
 * we pay it once and reuse the sessions. OrtSession.run() is safe to call from several
 * threads at once, so one shared instance serves every request.
 */
// @Component: Spring creates exactly one OnnxModels (a "singleton bean") at startup and
// hands it to any class that asks for it in its constructor.
@Component
public class OnnxModels implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(OnnxModels.class);

    // The ONNX Runtime engine: one per JVM, shared by all sessions.
    private final OrtEnvironment env;
    // A "session" = one loaded model, ready to run. Like a prepared JDBC statement:
    // set up once, executed many times with different inputs.
    private final OrtSession detector;
    private final OrtSession classifier;

    /**
     * @param modelsDir the carid.models-dir value from application.properties. @Value is
     *                  how Spring injects a config value into a constructor parameter.
     */
    public OnnxModels(@Value("${carid.models-dir}") Path modelsDir) throws OrtException {
        // toAbsolutePath() so the log and any error show exactly where we looked.
        Path dir = modelsDir.toAbsolutePath().normalize();
        log.info("Loading ONNX models from {}", dir);

        env = OrtEnvironment.getEnvironment();
        detector = openSession(dir.resolve("detector.onnx"));
        classifier = openSession(dir.resolve("car_model.onnx"));

        // Print each model's inputs and outputs. These must match preprocessing.json
        // ("image" for the detector, "input"/"logits" for the classifier); if training
        // ever changes them, this log is where you'd notice first.
        describe("detector", detector);
        describe("classifier", classifier);
    }

    private OrtSession openSession(Path file) throws OrtException {
        // Fail fast with a clear message instead of a cryptic native error later.
        if (!Files.isRegularFile(file)) {
            throw new IllegalStateException("Model file not found: " + file
                    + " (run training/export_onnx.py, or set carid.models-dir)");
        }
        long start = System.nanoTime();
        OrtSession session = env.createSession(file.toString(), new OrtSession.SessionOptions());
        log.info("Loaded {} in {} ms", file.getFileName(), (System.nanoTime() - start) / 1_000_000);
        return session;
    }

    private static void describe(String name, OrtSession session) throws OrtException {
        for (Map.Entry<String, NodeInfo> in : session.getInputInfo().entrySet()) {
            log.info("{} input  '{}': {}", name, in.getKey(), in.getValue().getInfo());
        }
        for (Map.Entry<String, NodeInfo> out : session.getOutputInfo().entrySet()) {
            log.info("{} output '{}': {}", name, out.getKey(), out.getValue().getInfo());
        }
    }

    public OrtEnvironment env() {
        return env;
    }

    public OrtSession detector() {
        return detector;
    }

    public OrtSession classifier() {
        return classifier;
    }

    /**
     * Frees the models' native memory when the server shuts down. ONNX Runtime is C++
     * underneath, and the JVM's garbage collector can't free memory it didn't allocate,
     * so sessions must be closed explicitly. @PreDestroy makes Spring call this on Ctrl+C.
     */
    @PreDestroy
    @Override
    public void close() throws OrtException {
        detector.close();
        classifier.close();
    }
}
