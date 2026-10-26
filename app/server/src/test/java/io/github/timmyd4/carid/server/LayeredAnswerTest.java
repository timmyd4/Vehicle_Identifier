package io.github.timmyd4.carid.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import io.github.timmyd4.carid.server.LayeredAnswer.Level;
import org.junit.jupiter.api.Test;

/**
 * Checks each place LayeredAnswer can stop, using a tiny made-up label list and
 * hand-picked probabilities instead of the real model.
 *
 * <p>Why made-up numbers: with the real model we can't choose what it outputs, so we could
 * never force, say, "sure of the generation but torn on the facelift". Here every case is
 * one line of numbers, and we can work out the right answer by hand. A plain unit test:
 * no models, no Spring, runs in milliseconds.
 */
class LayeredAnswerTest {

    private static final float THRESHOLD = 0.7f;

    // Six classes. Position in this array = class index, like labels.json.
    private static final CarLabel[] LABELS = {
            label(0, "honda", "civic", "8th gen", "pre", 2006, 2008),
            label(1, "honda", "civic", "8th gen", "facelift", 2009, 2011),
            label(2, "honda", "civic", "7th gen", "pre", 2001, 2003),
            label(3, "honda", "accord", "7th gen", "single", 2003, 2007),
            label(4, "acura", "csx", "1st gen", "single", 2006, 2011),
            label(5, "toyota", "corolla", "E140", "single", 2009, 2013),
    };

    @Test
    void sureOfGenerationButTornOnFacelift() {
        // Honda 0.89, Civic 0.83, 8th gen 0.75, but facelift alone only 0.45.
        LayeredAnswer a = decide(0.30f, 0.45f, 0.08f, 0.06f, 0.06f, 0.05f);

        assertEquals(Level.GENERATION, a.level());
        assertEquals("Honda Civic, 8th gen (2006-2011)", a.text());  // years span BOTH looks
        assertEquals(0.75f, a.confidence(), 1e-6);
        assertNull(a.phase(), "it isn't sure of the look, so it mustn't claim one");
    }

    @Test
    void sureOfTheFacelift() {
        LayeredAnswer a = decide(0.05f, 0.80f, 0.05f, 0.04f, 0.03f, 0.03f);

        assertEquals(Level.LOOK, a.level());
        assertEquals("Honda Civic, 8th gen, facelift (2009-2011)", a.text());
        assertEquals("facelift", a.phase());
    }

    @Test
    void sureOfModelButNotGeneration() {
        // 8th gen 0.40 vs 7th gen 0.40: Civic for sure (0.80), generation a coin flip.
        LayeredAnswer a = decide(0.20f, 0.20f, 0.40f, 0.10f, 0.05f, 0.05f);

        assertEquals(Level.MODEL, a.level());
        assertEquals("Honda Civic", a.text());
        assertNull(a.generation());
        assertNull(a.yearStart());
    }

    @Test
    void sureOfMakeOnly() {
        // Civic 0.45 vs Accord 0.40: Honda for sure (0.85), model unclear.
        LayeredAnswer a = decide(0.20f, 0.20f, 0.05f, 0.40f, 0.10f, 0.05f);

        assertEquals(Level.MAKE, a.level());
        assertEquals("Honda", a.text());
        assertEquals("honda", a.make());
        assertNull(a.model());
    }

    @Test
    void notSureWhenBeliefIsSpreadOut() {
        // What an unknown car tends to look like: no make gets near the threshold.
        LayeredAnswer a = decide(0.15f, 0.10f, 0.10f, 0.10f, 0.30f, 0.25f);

        assertEquals(Level.NOT_SURE, a.level());
        assertEquals("Not sure", a.text());
        assertEquals(0.45f, a.confidence(), 1e-6);  // the best make (Honda) still gets reported
        assertNull(a.make());
    }

    @Test
    void generationWithoutFaceliftIsTheFullAnswer() {
        LayeredAnswer a = decide(0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.95f);

        assertEquals(Level.GENERATION, a.level());
        assertEquals("Toyota Corolla, E140 (2009-2013)", a.text());
    }

    @Test
    void summingSiblingsBeatsTheBestSingleClass() {
        // The key idea: NO class reaches 0.7 on its own, yet the answer is confident,
        // because the belief is split between two looks of the same generation.
        LayeredAnswer a = decide(0.38f, 0.38f, 0.06f, 0.06f, 0.06f, 0.06f);

        assertEquals(Level.GENERATION, a.level());
        assertEquals(0.76f, a.confidence(), 1e-6);
    }

    private static LayeredAnswer decide(float... probs) {  // float... = any number of floats, as an array
        return LayeredAnswer.decide(LABELS, probs, THRESHOLD);
    }

    /** A label with readable names derived the simple way (good enough for these test cars). */
    private static CarLabel label(int index, String make, String model, String generation,
                                  String phase, int yearStart, int yearEnd) {
        String makeName = capitalize(make);
        String carName = makeName + " " + capitalize(model);
        String years = yearStart + "-" + yearEnd;
        return new CarLabel(index, make + "_" + model + "_" + years, carName + " (" + years + ")",
                makeName, carName, make, model, yearStart, yearEnd, generation, phase);
    }

    private static String capitalize(String word) {
        return Character.toUpperCase(word.charAt(0)) + word.substring(1);
    }
}
