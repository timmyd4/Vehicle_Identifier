# Maven Notes

What I learned about Maven while building the Java side of the Car Identifier (`app/`).
Each section has the concept, why it matters, and how to see it for yourself.

---

## 1. What Maven is

A build tool and dependency manager for Java. You describe the project in `pom.xml`
(Project Object Model), and Maven downloads libraries, compiles, tests and packages it.
It works **declaratively**: you say *what* the project is, not the steps to build it.

## 2. Coordinates

Every project and library is identified by three values (together called a GAV):

| Part | Meaning | Ours |
|---|---|---|
| `groupId` | Who owns it, as a reversed domain | `io.github.timmyd4` |
| `artifactId` | The name of this artifact | `car-identifier-parent` |
| `version` | Which version | `0.1.0-SNAPSHOT` |

- No domain? Use `io.github.<github-username>`, since GitHub gives you `<username>.github.io`.
- `-SNAPSHOT` = work in progress, can change. A release version (`0.1.0`) never changes once published.
- Modules inherit `groupId` and `version` from the parent, so they only declare `artifactId`.

## 3. Packaging

`<packaging>` decides what the build produces. Default is `jar`.
- `pom`: produces no jar. Used for parent/aggregator projects that only hold shared config.
- `jar`: compiled classes in a jar (our server and client modules will be this).

The packaging type also decides **which plugin goals run in each lifecycle phase** (see section 10).

## 4. The Super POM (convention over configuration)

Every POM silently inherits from a built-in Super POM, like every Java class extends `Object`.
It supplies defaults:
- Maven Central as the repository
- Source in `src/main/java`, tests in `src/test/java`, resources in `src/main/resources`
- Build output in `target/` (classes in `target/classes`)

Follow the standard layout and you configure almost nothing.

## 5. Properties

Named values, used as `${name}`. Like constants.

```xml
<properties>
    <spring-boot.version>4.1.2</spring-boot.version>
</properties>
...
<version>${spring-boot.version}</version>
```

Some names are read directly by plugins:
- `maven.compiler.release`: Java version to compile for (we use 17 because GWT supports up to 17)
- `project.build.sourceEncoding`: file encoding (UTF-8, otherwise Windows uses cp1252)

## 6. Parent POMs (inheritance)

A parent POM holds config that child modules inherit: versions, plugin setup, properties.
Like an abstract base class. **A POM can have only one parent.**

## 7. `dependencyManagement` vs `dependencies`

The most important distinction:

| Section | Meaning | Adds to classpath? |
|---|---|---|
| `<dependencies>` | "This project USES this library" | Yes |
| `<dependencyManagement>` | "IF anyone uses this library, use THIS version" | No |

The parent pins versions in `dependencyManagement`. Each module lists only what it needs
in `dependencies`, **with no version**. Every module ends up on the same versions.

## 8. BOM imports

A BOM (bill of materials) is a POM that contains only a `dependencyManagement` list.
Importing one copies its whole version list into yours:

```xml
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-dependencies</artifactId>
            <version>4.1.2</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>
```

- **Why not inherit `spring-boot-starter-parent`?** Our modules' one parent slot is our own POM.
  Importing the BOM gives the same version management without using that slot.
- **Catch:** a BOM manages *dependency* versions only, not *plugin* versions.

## 9. `pluginManagement`

Same idea as `dependencyManagement`, but for plugins: pins versions and shared settings
without activating anything.

**Why pin core plugins?** Maven has built-in default plugin versions, but they depend on
which Maven version you run. Pinning makes the build behave the same everywhere.
We found two we'd missed (deploy, site) by reading the effective POM.

## 10. Plugins, goals and the lifecycle

Maven itself does almost nothing. **Plugins** do the work, through **goals**
(`compiler:compile`, `surefire:test`, `jar:jar`).

The **default lifecycle** is an ordered list of phases:

```
validate → compile → test → package → verify → install → deploy
```

- `mvn <phase>` runs **every phase up to and including** that one. `mvn package` = validate + compile + test + package.
- Each phase has plugin goals **bound** to it, depending on packaging:
  - `jar`: compile → `compiler:compile`, test → `surefire:test`, package → `jar:jar`, ...
  - `pom`: only `install` and `deploy`. Nothing to compile.
- Separate lifecycles: `clean` (deletes `target/`) and `site` (docs website). Hence `mvn clean package`.

| Command | What it does | Creates `target/`? |
|---|---|---|
| `mvn validate` | Checks the POM is valid | No |
| `mvn compile` | Compiles to `target/classes` | Yes |
| `mvn package` | Compiles, tests, builds the jar | Yes |
| `mvn install` | ...and copies the jar into `~/.m2` | Yes |
| `mvn clean` | Deletes `target/` | (removes it) |

## 11. `target/` and the local repository

- **`target/`**: disposable build output, one per module. Never committed (`.gitignore`).
  Source plus `pom.xml` is the recipe; `target/` is the meal.
- **`~/.m2/repository`**: the local repository, a cache of every downloaded library shared
  by all projects on the machine. Download once, reuse everywhere. That's why a second
  run is fast and shows no `Downloading` lines.

## 12. Modules, aggregation and the reactor

A multi-module project is one parent folder with several sub-projects, built by one command.

```
app/pom.xml            <modules><module>client</module></modules>
app/client/pom.xml     <parent>car-identifier-parent</parent>
```

Two separate ideas, usually in the same POM:

| | Declared in | Direction | Meaning |
|---|---|---|---|
| **Aggregation** | parent's `<modules>` | parent → child | "Build these along with me" |
| **Inheritance** | child's `<parent>` | child → parent | "Copy my config from there" (like `extends`) |

- The set of projects built together is the **reactor**. Maven sorts it so anything a
  module depends on is built first (`Reactor Build Order` in the output).
- `Reactor Summary` at the end shows SUCCESS / FAILURE / SKIPPED per module.
- `<parent>` finds the parent at `../pom.xml` by default (`<relativePath>`).

## 13. Plugin-supplied packaging types

Plugins can add new packaging types. Our client uses `<packaging>gwt-app</packaging>`,
which comes from `gwt-maven-plugin` declared with `<extensions>true</extensions>`.
That packaging binds GWT goals into the lifecycle, so plain `mvn package` also runs the
Java-to-JavaScript compiler.

Also: listing a plugin under `<build><plugins>` **activates** it. Listing it under
`<pluginManagement>` only pins its version/settings.

## 14. Reading build output

Each plugin goal that runs prints a line:

```
--- gwt:1.3.0:compile (default-compile) @ car-identifier-client ---
    prefix:version:goal (execution id)   @ module
```

- `default-...` execution ids = bound automatically by the packaging type.
- **Run phases, not goals.** `mvn package` runs everything up to package in order.
  `mvn gwt:compile` runs only that goal and skips javac etc. Direct goals are for
  diagnostics like `help:effective-pom`.
- `[debug release 17]` on the compiler line confirms `maven.compiler.release` was applied.
- Transitive dependencies: when a library or plugin needs other libraries, Maven downloads
  those too (`.pom` first to learn the dependencies, then the `.jar`).

## 15. Libraries that aren't on Maven Central (SmartGWT)

Maven resolves a dependency by checking `~/.m2/repository` **first**, then remote repositories.
So a library that isn't published anywhere can still be used: put it into `~/.m2` yourself.

SmartGWT LGPL is a zip download, not a Maven artifact. Isomorphic's plugin converts it:

```bash
# run OUTSIDE a multi-module project, or the goal runs once per module
mvn com.isomorphic:isc-maven-plugin:1.5.5:install -Dproduct=SMARTGWT -Dlicense=LGPL -DbuildNumber=14.1p
```

1. downloads `smartgwt-14.1p.zip` from smartclient.com
2. unpacks it, writes a POM for each jar
3. **installs** them into `~/.m2/repository/com/isomorphic/...` (version `14.1-p20261024`)

- One-time per machine. Builds afterwards never run the plugin; they just read `~/.m2`.
- Re-run on a new machine, to upgrade, or if `~/.m2` is wiped
  (symptom: `Could not find artifact com.isomorphic...`).
- Calling a plugin not declared in any POM needs **full coordinates**: `groupId:artifactId:version:goal`.
- `-Dname=value` passes settings to the plugin. A typo (`-Dlicense-LGPL`) gives
  "parameters 'license' ... are missing or invalid".

**Local repository layout:** `groupId` dots become folders, then artifactId, then version:
`~/.m2/repository/org/gwtproject/gwt-user/2.13.1/gwt-user-2.13.1.jar` (+ `.pom`, `.sha1` checksums).

## 16. `mvn dependency:tree` and transitive dependencies

Shows every library a module ends up with, and **who pulled it in**:

```
+- org.gwtproject:gwt-user:jar:2.13.1:compile        <- we declared this
|  +- com.google.jsinterop:jsinterop-annotations...  <- gwt-user's POM declared this (transitive)
\- com.isomorphic.smartgwt.lgpl:smartgwt-lgpl:...    <- empty POM, so nothing underneath
```

A library's POM lists its own dependencies; Maven follows them recursively.

## 17. Lesson: a BOM in the parent applies to EVERY module

The tree showed Jetty `9.4.58` (what GWT asked for) mixed with Jetty `12.1.14` in the client.
Why: the Spring Boot BOM was imported in the **parent**, so its version rules applied to the
client too and overrode GWT's transitive versions. It didn't break the GWT compile, but it would
break GWT dev mode (which runs on Jetty 9.4).

**Rule:** import a BOM only in the modules that need it. Spring Boot's BOM belongs in the server
module, not the shared parent. Always check `dependency:tree` after adding a BOM.

**The fix we made:** the Spring Boot BOM moved from the parent to `server/pom.xml`. A module's own
`dependencyManagement` is **merged** with the inherited one, so the server gets Spring's versions
plus the parent's (onnxruntime, SmartGWT), and the client only gets the parent's.
Afterwards: client = all Jetty 9.4, server = all Jetty 12.1.

## 18. Starters and exclusions (Tomcat → Jetty)

**Starters** (`spring-boot-starter-*`) contain almost no code. Each is a curated bundle of
dependencies for one job. `spring-boot-starter-webmvc` = Spring MVC + Jackson + an embedded server.

**Exclusions** cut one branch out of the dependency tree:

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-webmvc</artifactId>
    <exclusions>
        <exclusion>  <!-- "everything you normally bring, EXCEPT Tomcat" -->
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-tomcat</artifactId>
        </exclusion>
    </exclusions>
</dependency>
<dependency>  <!-- ...and Jetty instead -->
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-jetty</artifactId>
</dependency>
```

- No version in an exclusion: it removes the artifact whatever its version.
- Spring Boot **auto-configuration** starts whichever server it finds on the classpath, so the
  swap is POM-only. Proof at runtime: `Jetty started on port 8080`.
- Read full artifactIds: `tomcat-embed-el` is an expression-language library that the Tomcat
  project publishes, not the Tomcat server (`tomcat-embed-core`).

## 19. Fat jars (spring-boot-maven-plugin)

A normal jar holds only your classes, so `java -jar` fails with `ClassNotFoundException`.
The `repackage` goal (bound to `package`, runs after `jar:jar`) rebuilds it with every
dependency in `BOOT-INF/lib/` plus a launcher. The small original is kept as `*.jar.original`.

```bash
java -jar server/target/car-identifier-server-0.1.0-SNAPSHOT.jar
```

## 20. Building part of the reactor: `-pl` and `-am`

- `mvn package -pl server` builds **only** the listed module(s) (comma-separated).
- `-am` ("also make") adds the modules they depend on: `mvn package -pl server -am` builds
  client + server. Without `-am`, Maven looks for the client's war in `~/.m2`, where it was
  never installed, and fails to resolve it.
- Run ONE test class: `mvn -pl server -am test -Dtest=LayeredAnswerTest -Dsurefire.failIfNoSpecifiedTests=false`
  - `-Dtest=` applies to every module in the build, and the client has no such test.
    `failIfNoSpecifiedTests=false` stops that from failing the build.
  - The trap (hit 2026-10-26): the same flag also hides a TYPO. `-Dtest=LayeredAnswerText`
    ran zero tests and still said BUILD SUCCESS. Always check for the `Tests run: N` line.
  - `test` is fast even with `-am`: GWT's Java -> JavaScript compile runs in `package`, after `test`.

## 21. Inter-module dependencies, scopes, and unpacking (how the UI gets into the server jar)

The server declares the client as a dependency:

```xml
<dependency>
    <groupId>io.github.timmyd4</groupId>
    <artifactId>car-identifier-client</artifactId>
    <version>${project.version}</version>
    <type>war</type>              <!-- the client's output is a .war, not a jar -->
    <scope>provided</scope>       <!-- don't pack the war itself into the fat jar -->
    <exclusions>                  <!-- don't inherit the client's GWT/SmartGWT/Jetty 9.4 deps -->
        <exclusion><groupId>*</groupId><artifactId>*</artifactId></exclusion>
    </exclusions>
</dependency>
```

- **Build order:** the reactor now builds client before server.
- **Scopes** say where a dependency is used:

| Scope | Compile | Tests | Packaged / at run time |
|---|---|---|---|
| `compile` (default) | yes | yes | yes |
| `provided` | yes | yes | no ("someone else provides it") |
| `runtime` | no | yes | yes |
| `test` | no | yes | no |

- **Unpacking:** `maven-dependency-plugin:unpack-dependencies`, bound to `process-resources`,
  extracts `app/**` from the war into `target/classes/static/`. `jar:jar` then zips it, and
  Spring Boot serves anything in `static/` (`index.html` for `/`).
- An `<execution>` with an `<id>` binds a plugin goal to a phase; the id appears in build output.

## 22. Properties can be overridden with `-D`

```xml
<gwt.style>PRETTY</gwt.style>     <!-- client/pom.xml -->
<style>${gwt.style}</style>       <!-- plugin config -->
```
`mvn package -Dgwt.style=OBFUSCATED` overrides it for one build: command line beats `<properties>`.

## 23. Diagnostic commands

| Command | Use it to |
|---|---|
| `mvn help:effective-pom` | See the final merged POM: Super POM defaults, imported BOMs, resolved properties, bound plugins |
| `mvn help:effective-pom \| grep -A1 <name>` | Answer "what version/setting is Maven *actually* using for X?" |
| `mvn dependency:tree` | See every dependency per module and which one pulled it in; spot version conflicts |

---

## Phase 3 recap: one command, one jar

```
mvn clean package          (from app/)
  parent  [pom]      nothing to build; shares versions + plugin pins
  client  [gwt-app]  App.java --javac--> .class --GWT--> JavaScript --> client .war
  server  [jar]      unpack war's app/ into static/ + compile + jar + repackage --> fat jar

java -jar server/target/car-identifier-server-0.1.0-SNAPSHOT.jar   → http://localhost:8080
```

*Next (Phase 4): test-scope dependencies (JUnit via spring-boot-starter-test), ONNX Runtime in the server.*
