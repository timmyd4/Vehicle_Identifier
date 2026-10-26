package io.github.timmyd4.carid.server;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * The verdict for one vehicle: the most detailed answer the model is sure enough about.
 *
 * <p>THE BIG PICTURE. The classifier is ONE model with one flat list of classes. Each class
 * is a full answer ("Honda Civic, 8th gen, facelift"). It never decides "Honda first, then
 * Civic"; it spreads its belief over all classes at once. But the classes form layers:
 *
 * <pre>
 *   make         Honda                        = sum of every Honda class        86%
 *    model        Civic                       = sum of every Civic class        83%
 *     generation   8th gen                    = sum of its pre + facelift       75%
 *      look         facelift (2009-2011)      = one class                       45%
 * </pre>
 *
 * <p>Going down, each layer is part of the one above, so its probability can only stay the
 * same or shrink. We walk down, always taking the most likely group inside the previous
 * winner, and STOP at the first layer below the threshold. With a 70% threshold the example
 * above answers "Honda Civic, 8th gen (2006-2011)": sure of the generation, torn on the
 * facelift. Like a person saying "it's an 8th-gen Civic, can't tell which year from here."
 *
 * <p>If even the make is below the threshold, the answer is "Not sure". That's the honest
 * reply to a car the model has never seen (open-set recognition): it must pick one of its
 * classes, so an unknown car shows up as belief spread thinly over many of them.
 *
 * <p>Like CarResult, this record is also the JSON the server sends; Jackson writes the enum
 * as its name ("GENERATION") and null fields as null.
 *
 * @param level      how deep the answer goes
 * @param text       what to show a person, e.g. "Honda Civic, 8th gen (2006-2011)"
 * @param confidence the probability of the deepest layer reached (0-1)
 * @param make       e.g. "honda"; null when NOT_SURE
 * @param model      e.g. "civic"; null above MODEL
 * @param generation e.g. "8th gen"; null above GENERATION
 * @param phase      e.g. "facelift"; null above LOOK
 * @param yearStart  first model year covered; null above GENERATION
 * @param yearEnd    last model year covered; null above GENERATION
 */
public record LayeredAnswer(Level level, String text, float confidence,
                            String make, String model, String generation, String phase,
                            Integer yearStart, Integer yearEnd) {

    /** How far down the layers the answer goes. Ordered from vaguest to most detailed. */
    public enum Level {
        NOT_SURE, MAKE, MODEL, GENERATION,
        /** The exact class: generation plus pre-facelift/facelift. */
        LOOK
    }

    /**
     * Walk down the layers and stop at the first one below {@code threshold}.
     *
     * @param labels    what each class position means (from labels.json)
     * @param probs     the classifier's probabilities, same positions, adding up to 1
     * @param threshold how sure a layer must be to be reported, e.g. 0.7
     */
    static LayeredAnswer decide(CarLabel[] labels, float[] probs, float threshold) {
        // Start with every class, then narrow down to the winner's classes at each layer.
        List<Integer> all = IntStream.range(0, labels.length).boxed().toList();

        Group make = bestGroup(labels, probs, all, CarLabel::make);
        if (make.probability() < threshold) {
            return new LayeredAnswer(Level.NOT_SURE, "Not sure", make.probability(),
                    null, null, null, null, null, null);
        }

        Group model = bestGroup(labels, probs, make.classes(), CarLabel::model);
        if (model.probability() < threshold) {
            return answer(Level.MAKE, labels, make);
        }

        Group generation = bestGroup(labels, probs, model.classes(), CarLabel::generation);
        if (generation.probability() < threshold) {
            return answer(Level.MODEL, labels, model);
        }

        // A generation that never had a facelift is a single class: the generation IS the
        // whole answer, so there's no look to decide between.
        if (!labels[generation.classes().get(0)].hasFacelifts()) {
            return answer(Level.GENERATION, labels, generation);
        }

        // Last layer: each class is its own group (grouping by the unique class name).
        Group look = bestGroup(labels, probs, generation.classes(), CarLabel::className);
        if (look.probability() < threshold) {
            return answer(Level.GENERATION, labels, generation);
        }
        return answer(Level.LOOK, labels, look);
    }

    /**
     * Some classes, and the sum of their probabilities. classes are positions in labels.
     */
    private record Group(List<Integer> classes, float probability) {
    }

    /**
     * Among {@code classes}, add up the probabilities per group (e.g. per make) and return
     * the group with the biggest total.
     *
     * <p>{@code groupOf} says which group a class belongs to. Passing CarLabel::make groups by
     * make, CarLabel::model by model, and so on, so one method serves every layer. (A
     * Function is Java's type for "a method you pass around", like a one-method interface.)
     *
     * <p>Grouping by model only ever looks inside one make, and by generation inside one
     * model, so equal names in different cars ("2nd gen") never get mixed up.
     */
    private static Group bestGroup(CarLabel[] labels, float[] probs, List<Integer> classes,
                                   Function<CarLabel, String> groupOf) {
        Map<String, Double> totals = new HashMap<>();
        Map<String, List<Integer>> members = new HashMap<>();
        for (int i : classes) {
            String group = groupOf.apply(labels[i]);
            // merge(): put the value if the key is new, otherwise combine with Double::sum.
            totals.merge(group, (double) probs[i], Double::sum);
            members.computeIfAbsent(group, key -> new ArrayList<>()).add(i);
        }

        String best = null;
        for (Map.Entry<String, Double> entry : totals.entrySet()) {
            if (best == null || entry.getValue() > totals.get(best)) {
                best = entry.getKey();
            }
        }
        return new Group(members.get(best), totals.get(best).floatValue());
    }

    /** Build the answer for a group reached at {@code level}, filling only what that level knows. */
    private static LayeredAnswer answer(Level level, CarLabel[] labels, Group group) {
        CarLabel first = labels[group.classes().get(0)];  // all classes in the group share the upper layers

        boolean knowsModel = level.compareTo(Level.MODEL) >= 0;  // enums compare by declaration order
        boolean knowsGeneration = level.compareTo(Level.GENERATION) >= 0;
        boolean knowsLook = level == Level.LOOK;

        // A generation's years run from its first look's start to its last look's end.
        Integer yearStart = null;
        Integer yearEnd = null;
        if (knowsGeneration) {
            yearStart = Integer.MAX_VALUE;
            yearEnd = Integer.MIN_VALUE;
            for (int i : group.classes()) {
                yearStart = Math.min(yearStart, labels[i].yearStart());
                yearEnd = Math.max(yearEnd, labels[i].yearEnd());
            }
        }

        String text = switch (level) {
            case MAKE -> first.makeName();
            case MODEL -> first.carName();
            case GENERATION -> first.carName() + ", " + first.generation()
                    + " (" + yearStart + "-" + yearEnd + ")";
            case LOOK -> first.carName() + ", " + first.generation() + ", " + phaseText(first.phase())
                    + " (" + yearStart + "-" + yearEnd + ")";
            case NOT_SURE -> "Not sure";
        };

        return new LayeredAnswer(level, text, group.probability(),
                first.make(),
                knowsModel ? first.model() : null,
                knowsGeneration ? first.generation() : null,
                knowsLook ? first.phase() : null,
                yearStart, yearEnd);
    }

    /** "pre" -> "pre-facelift", "facelift2" -> "2nd facelift", and so on. */
    private static String phaseText(String phase) {
        return switch (phase) {
            case "pre" -> "pre-facelift";
            case "facelift" -> "facelift";
            case "facelift2" -> "2nd facelift";
            case "facelift3" -> "3rd facelift";
            default -> phase;
        };
    }
}
