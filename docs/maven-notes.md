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

## 12. Diagnostic commands

| Command | Use it to |
|---|---|
| `mvn help:effective-pom` | See the final merged POM: Super POM defaults, imported BOMs, resolved properties, bound plugins |

---

*More sections to come: modules and reactor builds, dependency scopes, `dependency:tree`, `-pl`/`-am`, ...*
