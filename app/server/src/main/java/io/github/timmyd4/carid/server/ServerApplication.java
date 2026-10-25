package io.github.timmyd4.carid.server;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Starts the Car Identifier server.
 *
 * <p>Big picture: this one class is all Spring Boot needs to boot a web server.
 * Later phases add controllers (POST /identify) and the ONNX model classes next to it.
 */
// @SpringBootApplication combines three things:
//   @Configuration       - this class can define beans (objects Spring creates and manages)
//   @EnableAutoConfiguration - Spring Boot looks at the classpath and configures what it finds
//                          (Jetty jars present -> start embedded Jetty; Jackson present -> JSON)
//   @ComponentScan       - find @Controller/@Service classes in this package and below
@SpringBootApplication
public class ServerApplication {

    // A plain main() method: the fat jar's launcher ends up calling this.
    // SpringApplication.run() creates the Spring context, starts Jetty, and blocks
    // until the server is stopped (Ctrl+C).
    public static void main(String[] args) {
        SpringApplication.run(ServerApplication.class, args);
    }
}
