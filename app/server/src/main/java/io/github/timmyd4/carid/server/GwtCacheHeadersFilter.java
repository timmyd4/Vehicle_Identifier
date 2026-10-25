package io.github.timmyd4.carid.server;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/**
 * Tells browsers how to cache the GWT UI files, so a rebuilt UI shows up on a normal reload.
 *
 * <p>GWT names its files by a convention, but browsers only follow HTTP headers, and Spring
 * Boot doesn't know the convention. Without this filter a browser can keep an old loader
 * after a rebuild and keep running the old UI.
 * <ul>
 *   <li>{@code *.nocache.js}: the small loader. Must always be re-checked, because it names
 *       the current compiled files.</li>
 *   <li>{@code *.cache.js}: compiled code named by a hash of its contents. A new build gives
 *       new names, so these can be cached for a year.</li>
 * </ul>
 */
// @Component: Spring finds this class (component scan) and Spring Boot registers it as a
// servlet filter, so it sees every request before the static-file handler serves it.
@Component
public class GwtCacheHeadersFilter extends OncePerRequestFilter {

    private static final String ONE_YEAR = "public, max-age=31536000, immutable";

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
                                    FilterChain chain) throws ServletException, IOException {
        String path = request.getRequestURI();

        // Check nocache first: "app.nocache.js" also ends with ".cache.js".
        if (path.endsWith(".nocache.js")) {
            // no-cache = "you may keep a copy, but ask the server if it changed before using it"
            response.setHeader("Cache-Control", "no-cache");
        } else if (path.endsWith(".cache.js")) {
            response.setHeader("Cache-Control", ONE_YEAR);
        }

        // Pass the request on, so the static-file handler still serves the file.
        chain.doFilter(request, response);
    }
}
