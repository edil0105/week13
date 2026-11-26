// =====================================================================
//  КОМПЬЮТЕРЛІК ГРАФИКА — бір файлдық жоба
//
//  1-АПТА:    терезе, фон анимациясы, FPS
//  СЕМИНАР 4: uniform, dt, цикл "жаңарту (update)" және "сызу (draw)"
//  5-АПТА:    индекс буфері (EBO), glDrawElements
//  7-АПТА:    координаттар кеңістіктері, MVP
//  8-АПТА:    тереңдік буфері, back-face culling, бес куб
//  9-АПТА:    камера моделі, LookAt, пернетақтамен жылжу
//  10-АПТА:   толық еркін камера — тінтуір (yaw/pitch), дөңгелек (fov)
//  11-АПТА:   текстуралар — stb_image, UV, орау, сүзгі, mipmap
//  12-АПТА:   бірнеше текстура, текстура бірліктері, mix()
//  13-АПТА:   ЖАРЫҚТАНДЫРУ НЕГІЗДЕРІ — нормаль атрибуты (stride 5 → 8),
//             normalMatrix, lightPos, lightColor, ambient + diffuse (Ламберт)
//             - базалық:     нормальдар, FragPos/Normal, Фонг шейдингі
//             - 1-тапсырма:  1 — жарық көзі шеңбер бойымен жүреді
//             - 2-тапсырма:  2 — max(..., 0.0) алынып тасталады (ambient = 0.4)
//             - 3-тапсырма:  3 — mat3(model) (қате) / transpose(inverse) (дұрыс)
//             - 4-тапсырма:  O / L — ambientStrength 0.0 … 0.5 (× dt)
//             - 5-тапсырма:  5 — Гуро (төбелік шейдер) / Фонг (фрагмент)
//             - қосымша:     Қ1 H — жарық көзінің кубы, Қ2 6 — бағытталған жарық,
//                            Қ3 7 — түрлі-түсті жарық, Қ4 8 — екі жарық көзі
//
//  Басқару:
//      Тінтуір / дөңгелек      — қарау / fov
//      W A S D, Space, Ctrl    — жүру, Shift — жүгіру
//      ← → ↑ ↓, PgUp / PgDn    — жарық көзін жылжыту (орбита өшірулі кезде)
//      C                       — жарықты алдыңғы жақтың ортасына (0, 0, 1.2)
//      1 2 3 5                 — 1, 2, 3, 5-тапсырмалар (ауыстырып-қосқыш)
//      O / L, 4                — ambient үлкейту / кішірейту, 4 — 0.1-ге қайтару
//      H 6 7 8                 — Қ1 / Қ2 / Қ3 / Қ4
//      P                       — күйді консольге шығару
//      R                       — камера мен жарықты бастапқы күйге қайтару
//      I  K  M                 — invert Y / pitch шектеуі / тінтуір режимі
//      F12  ESC                — скриншот / шығу
//
//  Іске қосу:  ./build/bin/cg                — қалыпты режим
//              ./build/bin/cg --screenshots  — README-дегі скриншоттар
// =====================================================================

#include <glad/gl.h>      // МІНДЕТТІ: glad әрқашан GLFW-дан БҰРЫН
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifndef ASSET_DIR
#define ASSET_DIR "assets"
#endif
#ifndef SCREENSHOT_DIR
#define SCREENSHOT_DIR "screenshots"
#endif

int gWidth = 800, gHeight = 600;

// =====================================================================
//  10-АПТА: КАМЕРА
// =====================================================================
struct Camera {
    glm::vec3 pos;
    glm::vec3 front;
    glm::vec3 up;
};

// 13-апта: бірінші куб y бойынша 2.5 есе созылған (3-тапсырма) —
// толық көрінуі үшін камера сәл алысырақ тұрады
const Camera START_CAMERA = {
    glm::vec3(0.0f, 0.0f,  5.0f),
    glm::vec3(0.0f, 0.0f, -1.0f),
    glm::vec3(0.0f, 1.0f,  0.0f)
};

Camera freeCam = START_CAMERA;
glm::vec3& camPos   = freeCam.pos;
glm::vec3& camFront = freeCam.front;
glm::vec3& camUp    = freeCam.up;

const float START_YAW   = -90.0f;
const float START_PITCH =   0.0f;
const float START_FOV   =  45.0f;

float yaw   = START_YAW;
float pitch = START_PITCH;
float lastX = 400.0f;
float lastY = 300.0f;
bool  firstMouse = true;
float fov   = START_FOV;

const float FOV_MIN =  1.0f;
const float FOV_MAX = 90.0f;

bool pitchLimit = true;
bool invertY    = false;
// WSL-де (WSLg) курсорды ұстау дұрыс жұмыс істемейді — сүйреу режимінен бастаймыз
bool dragMode   = std::getenv("WSL_DISTRO_NAME") != nullptr;

float angle = 0.0f;   // кубтардың айналу бұрышы

// =====================================================================
//  13-АПТА: ЖАРЫҚ КҮЙІ
// =====================================================================
const glm::vec3 START_LIGHT(1.2f, 1.0f, 2.0f);        // 5-қадам
const glm::vec3 FACE_LIGHT (0.0f, 0.0f, 1.2f);        // 5-тапсырма: жақтың ортасына
const glm::vec3 SUN_DIRECTION(-0.2f, -1.0f, -0.3f);   // Қ2
const glm::vec3 WHITE(1.0f, 1.0f, 1.0f);
const glm::vec3 RED  (1.0f, 0.25f, 0.2f);              // Қ4: бірінші көз
const glm::vec3 BLUE (0.2f, 0.45f, 1.0f);              // Қ4: екінші көз

struct LightState {
    float ambient     = 0.1f;   // 4-тапсырма: ambientStrength
    bool  orbit       = false;  // 1-тапсырма
    bool  noMax       = false;  // 2-тапсырма: max(..., 0.0) ЖОҚ (әдейі қате)
    bool  wrongNormal = false;  // 3-тапсырма: mat3(model) (әдейі қате)
    bool  gouraud     = false;  // 5-тапсырма
    bool  showLamp    = true;   // Қ1
    bool  directional = false;  // Қ2
    bool  colorLight  = false;  // Қ3
    bool  twoLights   = false;  // Қ4
};

LightState ls;
glm::vec3  lightPos = START_LIGHT;
float      savedAmbient = 0.1f;   // 2-тапсырма өшірілгенде ambient қайтарылады

// 1-тапсырма: шеңбер бойымен (радиусы 1.5, биіктігі 1.0)
glm::vec3 orbitPos(float t) {
    return glm::vec3(1.5f * sin(t), 1.0f, 1.5f * cos(t));
}

// Қ3: уақыт бойынша өзгеретін түс
glm::vec3 rainbowColor(float t) {
    return glm::vec3(sin(t * 2.0f) * 0.5f + 0.5f,
                     sin(t * 0.7f) * 0.5f + 0.5f,
                     sin(t * 1.3f) * 0.5f + 0.5f);
}

// Қ4: екінші көз біріншінің x бойынша айнадағы бейнесі (оң ↔ сол)
glm::vec3 secondLightPos(const glm::vec3& p) {
    return glm::vec3(-p.x, p.y, p.z);
}

// =====================================================================
//  Callback-тер (10-апта)
// =====================================================================
void updateCameraFront() {
    glm::vec3 dir;
    dir.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    dir.y = sin(glm::radians(pitch));
    dir.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    camFront = glm::normalize(dir);
}

// Камераны берілген нүктеге бұру (скриншоттар үшін): front → yaw/pitch
void aimCamera(const glm::vec3& pos, const glm::vec3& target) {
    camPos = pos;
    glm::vec3 d = glm::normalize(target - pos);
    pitch = glm::degrees(asin(d.y));
    yaw   = glm::degrees(atan2(d.z, d.x));
    updateCameraFront();
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (dragMode && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
        lastX = (float)xpos;
        lastY = (float)ypos;
        firstMouse = false;
        return;
    }
    if (firstMouse) {
        lastX = (float)xpos;
        lastY = (float)ypos;
        firstMouse = false;
    }

    float xoffset = (float)xpos - lastX;
    float yoffset = lastY - (float)ypos;
    if (invertY) yoffset = -yoffset;
    lastX = (float)xpos;
    lastY = (float)ypos;

    const float sensitivity = 0.1f;
    yaw   += xoffset * sensitivity;
    pitch += yoffset * sensitivity;

    if (yaw >  360.0f) yaw -= 360.0f;
    if (yaw < -360.0f) yaw += 360.0f;

    if (pitchLimit) {
        if (pitch >  89.0f) pitch =  89.0f;
        if (pitch < -89.0f) pitch = -89.0f;
    }
    updateCameraFront();
}

void scroll_callback(GLFWwindow*, double, double yoffset) {
    fov -= (float)yoffset;
    if (fov < FOV_MIN) fov = FOV_MIN;
    if (fov > FOV_MAX) fov = FOV_MAX;
}

void framebuffer_size_callback(GLFWwindow*, int width, int height) {
    if (width == 0 || height == 0) return;
    gWidth  = width;
    gHeight = height;
    glViewport(0, 0, width, height);
}

void saveScreenshot(const std::string& path) {
    std::vector<unsigned char> pixels((size_t)gWidth * gHeight * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, gWidth, gHeight, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    stbi_flip_vertically_on_write(1);
    if (stbi_write_png(path.c_str(), gWidth, gHeight, 3, pixels.data(), gWidth * 3))
        std::cout << "Скриншот: " << path << "\n";
    else
        std::cerr << "Скриншот сақталмады: " << path << "\n";
}

// Бір рет басу: перне басылған СӘТТЕ ғана true
bool keyPressedOnce(GLFWwindow* window, int key) {
    static bool last[GLFW_KEY_LAST + 1] = {};
    bool now   = glfwGetKey(window, key) == GLFW_PRESS;
    bool fired = now && !last[key];
    last[key]  = now;
    return fired;
}

void applyMouseMode(GLFWwindow* window) {
    glfwSetInputMode(window, GLFW_CURSOR, dragMode ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
    firstMouse = true;
}

void resetCamera() {
    freeCam = START_CAMERA;
    yaw   = START_YAW;
    pitch = START_PITCH;
    fov   = START_FOV;
    updateCameraFront();
}

const char* yesNo(bool b) { return b ? "иә" : "жоқ"; }

void printState() {
    std::printf("camPos = (%.2f, %.2f, %.2f)  yaw = %.1f  pitch = %.1f  fov = %.1f\n",
                camPos.x, camPos.y, camPos.z, yaw, pitch, fov);
    std::printf("lightPos = (%.2f, %.2f, %.2f)  ambient = %.2f\n",
                lightPos.x, lightPos.y, lightPos.z, ls.ambient);
    std::printf("орбита: %s  max жоқ: %s  mat3(model): %s  Гуро: %s\n",
                yesNo(ls.orbit), yesNo(ls.noMax), yesNo(ls.wrongNormal), yesNo(ls.gouraud));
    std::printf("шам: %s  бағытталған: %s  түрлі-түсті: %s  екі көз: %s\n",
                yesNo(ls.showLamp), yesNo(ls.directional), yesNo(ls.colorLight),
                yesNo(ls.twoLights));
}

void toggle(bool& flag, const char* name) {
    flag = !flag;
    std::cout << name << ": " << (flag ? "ҚОСУЛЫ" : "ӨШІРУЛІ") << "\n";
}

// =====================================================================
//  Пернетақта — әр кадрда
// =====================================================================
void processInput(GLFWwindow* window, float dt) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // --- Жылжу (9–10-апта) ---
    float camSpeed = 2.5f * dt;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) camSpeed *= 2.0f;

    glm::vec3 right = glm::normalize(glm::cross(camFront, camUp));
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camPos += camSpeed * camFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camPos -= camSpeed * camFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camPos -= camSpeed * right;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camPos += camSpeed * right;
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        camPos += camSpeed * glm::vec3(0.0f, 1.0f, 0.0f);
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
        camPos -= camSpeed * glm::vec3(0.0f, 1.0f, 0.0f);

    // --- Жарық көзін қолмен жылжыту (базалық тексерудің 4-тармағы) ---
    if (!ls.orbit) {
        float lightSpeed = 2.0f * dt;
        if (glfwGetKey(window, GLFW_KEY_LEFT)      == GLFW_PRESS) lightPos.x -= lightSpeed;
        if (glfwGetKey(window, GLFW_KEY_RIGHT)     == GLFW_PRESS) lightPos.x += lightSpeed;
        if (glfwGetKey(window, GLFW_KEY_UP)        == GLFW_PRESS) lightPos.z -= lightSpeed;
        if (glfwGetKey(window, GLFW_KEY_DOWN)      == GLFW_PRESS) lightPos.z += lightSpeed;
        if (glfwGetKey(window, GLFW_KEY_PAGE_UP)   == GLFW_PRESS) lightPos.y += lightSpeed;
        if (glfwGetKey(window, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS) lightPos.y -= lightSpeed;
    }
    if (keyPressedOnce(window, GLFW_KEY_C)) {
        ls.orbit = false;
        lightPos = FACE_LIGHT;
        std::cout << "lightPos = (0, 0, 1.2) — алдыңғы жақтың ортасына (5-тапсырма)\n";
    }

    // --- 4-тапсырма: ambient. Үздіксіз әрекет — dt-ге көбейтеміз (4-апта) ---
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) ls.ambient += 0.3f * dt;
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) ls.ambient -= 0.3f * dt;
    if (ls.ambient < 0.0f) ls.ambient = 0.0f;
    if (ls.ambient > 0.5f) ls.ambient = 0.5f;
    if (keyPressedOnce(window, GLFW_KEY_4)) {
        ls.ambient = 0.1f;
        std::cout << "ambient = 0.1\n";
    }

    // --- Міндетті тапсырмалар ---
    if (keyPressedOnce(window, GLFW_KEY_1)) toggle(ls.orbit, "Жарық орбитада (1-тапсырма)");
    if (keyPressedOnce(window, GLFW_KEY_2)) {
        toggle(ls.noMax, "max(..., 0.0) алынып тасталды (2-тапсырма)");
        // Кеңес бойынша айырмашылық көрінуі үшін ambient уақытша 0.4
        if (ls.noMax) { savedAmbient = ls.ambient; ls.ambient = 0.4f; }
        else          { ls.ambient = savedAmbient; }
        std::printf("ambient = %.2f\n", ls.ambient);
    }
    if (keyPressedOnce(window, GLFW_KEY_3))
        toggle(ls.wrongNormal, "Normal = mat3(model) * aNormal, А нұсқасы (3-тапсырма)");
    if (keyPressedOnce(window, GLFW_KEY_5))
        toggle(ls.gouraud, "Гуро шейдингі (5-тапсырма)");

    // --- Қосымша ---
    if (keyPressedOnce(window, GLFW_KEY_H)) toggle(ls.showLamp,    "Жарық көзінің кубы (Қ1)");
    if (keyPressedOnce(window, GLFW_KEY_6)) toggle(ls.directional, "Бағытталған жарық (Қ2)");
    if (keyPressedOnce(window, GLFW_KEY_7)) toggle(ls.colorLight,  "Түрлі-түсті жарық (Қ3)");
    if (keyPressedOnce(window, GLFW_KEY_8)) toggle(ls.twoLights,   "Екі жарық көзі (Қ4)");

    // --- 10-аптадан қалған пернелер (pitch шектеуі L-ден K-ға көшті:
    //     L енді ambient-ті азайтады) ---
    if (keyPressedOnce(window, GLFW_KEY_I)) {
        invertY = !invertY;
        std::cout << "invertY = " << (invertY ? "true" : "false") << "\n";
    }
    if (keyPressedOnce(window, GLFW_KEY_P)) printState();
    if (keyPressedOnce(window, GLFW_KEY_R)) {
        resetCamera();
        ls = LightState();
        lightPos = START_LIGHT;
        std::cout << "Камера мен жарық бастапқы күйде\n";
    }
    if (keyPressedOnce(window, GLFW_KEY_K)) {
        pitchLimit = !pitchLimit;
        if (pitchLimit) {
            if (pitch >  89.0f) pitch =  89.0f;
            if (pitch < -89.0f) pitch = -89.0f;
            updateCameraFront();
        }
        std::cout << "pitch шектеуі: " << (pitchLimit ? "ҚОСУЛЫ" : "ӨШІРУЛІ") << "\n";
    }
    if (keyPressedOnce(window, GLFW_KEY_M)) {
        dragMode = !dragMode;
        applyMouseMode(window);
        std::cout << "Тінтуір режимі: " << (dragMode ? "сүйреу (ЛКМ басулы)" : "FPS") << "\n";
    }
    if (keyPressedOnce(window, GLFW_KEY_F12)) {
        static int shot = 1;
        saveScreenshot(std::string(SCREENSHOT_DIR) + "/shot-" + std::to_string(shot++) + ".png");
    }
}

// =====================================================================
//  ШЕЙДЕРЛЕР
// =====================================================================

// Жарық есебі — Фонг (фрагмент) және Гуро (төбелік) шейдерлеріне ортақ.
// Ішіндегі формула дәрістегідей:
//     ambient = ambientStrength * lightColor
//     diffuse = max(dot(norm, lightDir), 0.0) * lightColor
const char* lightingGlsl = R"glsl(
uniform vec3  lightPos;
uniform vec3  lightColor;
uniform float ambientStrength;   // 4-тапсырма (O / L)
uniform bool  useMax;            // 2-тапсырма: false — max алынып тасталған
uniform bool  directional;       // Қ2
uniform vec3  lightDirection;    // Қ2: жарық ТАРАЛАТЫН бағыт (күннен жерге)
uniform bool  twoLights;         // Қ4
uniform vec3  lightPos2;         // Қ4
uniform vec3  lightColor2;       // Қ4

float lambert(vec3 norm, vec3 lightDir) {
    float d = dot(norm, lightDir);          // екеуі де бірлік → cos θ
    return useMax ? max(d, 0.0) : d;        // теріс мәнді кесу
}

vec3 lighting(vec3 norm, vec3 fragPos) {
    // L — беттен жарық көзіне қарай
    vec3 lightDir = directional ? normalize(-lightDirection)
                                : normalize(lightPos - fragPos);

    vec3 ambientColor = twoLights ? 0.5 * (lightColor + lightColor2) : lightColor;
    vec3 ambient = ambientStrength * ambientColor;
    vec3 diffuse = lambert(norm, lightDir) * lightColor;
    vec3 result  = ambient + diffuse;

    if (twoLights)                           // Қ4: ambient + diffuse1 + diffuse2
        result += lambert(norm, normalize(lightPos2 - fragPos)) * lightColor2;
    return result;
}
)glsl";

// --- Фонг шейдингі (базалық): жарық ФРАГМЕНТ шейдерінде ---
const char* phongVertexSrc = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;
out vec2 vTexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;
uniform bool wrongNormal;   // 3-тапсырма

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));         // дүниелік кеңістік: тек model
    if (wrongNormal)
        Normal = mat3(model) * aNormal;              // А нұсқасы — қате
    else
        Normal = normalMatrix * aNormal;             // Б нұсқасы — дұрыс (CPU-да)
    vTexCoord = aTexCoord;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)glsl";

const char* phongFragmentHead = R"glsl(
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec2 vTexCoord;

out vec4 FragColor;

uniform sampler2D ourTexture;
)glsl";

const char* phongFragmentMain = R"glsl(
void main() {
    vec3 norm = normalize(Normal);   // интерполяция ұзындықты бұзады — қайта нормалау
    FragColor = vec4(lighting(norm, FragPos), 1.0)
              * texture(ourTexture, vTexCoord);
}
)glsl";

// --- 5-тапсырма: Гуро шейдингі — жарық ТӨБЕЛІК шейдерде ---
const char* gouraudVertexHead = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec3 aNormal;

out vec3 LightResult;
out vec2 vTexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;
uniform bool wrongNormal;
)glsl";

const char* gouraudVertexMain = R"glsl(
void main() {
    vec3 FragPos = vec3(model * vec4(aPos, 1.0));
    vec3 norm    = normalize(wrongNormal ? mat3(model) * aNormal
                                         : normalMatrix * aNormal);
    LightResult  = lighting(norm, FragPos);          // әр ТӨБЕГЕ бір рет
    vTexCoord    = aTexCoord;
    gl_Position  = projection * view * vec4(FragPos, 1.0);
}
)glsl";

const char* gouraudFragmentSrc = R"glsl(
#version 330 core
in vec3 LightResult;
in vec2 vTexCoord;
out vec4 FragColor;
uniform sampler2D ourTexture;
void main() {
    FragColor = vec4(LightResult, 1.0) * texture(ourTexture, vTexCoord);
}
)glsl";

// --- Қ1: жарық көзінің кубы — өте қарапайым шейдер ---
const char* lampVertexSrc = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)glsl";

const char* lampFragmentSrc = R"glsl(
#version 330 core
out vec4 FragColor;
uniform vec3 lampColor;      // ақ; Қ3, Қ4-те жарықтың өз түсі
void main() {
    FragColor = vec4(lampColor, 1.0);
}
)glsl";

// Еден-торы мен прицел үшін түсті шейдер (оларда UV та, нормаль да жоқ)
const char* colorVertexSrc = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 vColor;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    vColor = aColor;
}
)glsl";

const char* colorFragmentSrc = R"glsl(
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, 1.0);
}
)glsl";

GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, log);
        std::cerr << "Shader compile error:\n" << log << "\n";
    }
    return shader;
}

GLuint createShaderProgram(const std::string& vsSrc, const std::string& fsSrc) {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc.c_str());
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc.c_str());
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[1024];
        glGetProgramInfoLog(program, 1024, nullptr, log);
        std::cerr << "Program link error:\n" << log << "\n";
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

// =====================================================================
//  11-аптадағы loadTexture — СОЛ ҚАЛПЫНДА
// =====================================================================
unsigned int loadTexture(const char* path) {
    unsigned int id;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_set_flip_vertically_on_load(true);

    int w, h, n;
    unsigned char* data = stbi_load(path, &w, &h, &n, 0);
    if (data) {
        GLenum fmt = (n == 4) ? GL_RGBA : GL_RGB;
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, fmt, w, h, 0, fmt, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        std::cout << path << "  " << w << "x" << h << "  арна: " << n << std::endl;
    } else {
        std::cerr << "Сурет жүктелмеді: " << path << std::endl;
    }
    stbi_image_free(data);
    return id;
}

// =====================================================================
//  ГЕОМЕТРИЯ — 13-апта, 1-қадам: әр жолдың соңында НОРМАЛЬ (stride 8)
//  Куб жазық беттерден тұрады: бір жақтың алты төбесінде нормаль бірдей
//  (flat shading). Таңбасы — жақтың "сыртқа" қарауы бойынша.
// =====================================================================
const float cubeVertices[] = {
    //  поз                   UV            нормаль
    // алдыңғы жақ (z = +0.5) → (0, 0, 1)
    -0.5f, -0.5f,  0.5f,   0.0f, 0.0f,   0.0f,  0.0f,  1.0f,
     0.5f, -0.5f,  0.5f,   1.0f, 0.0f,   0.0f,  0.0f,  1.0f,
     0.5f,  0.5f,  0.5f,   1.0f, 1.0f,   0.0f,  0.0f,  1.0f,
    -0.5f, -0.5f,  0.5f,   0.0f, 0.0f,   0.0f,  0.0f,  1.0f,
     0.5f,  0.5f,  0.5f,   1.0f, 1.0f,   0.0f,  0.0f,  1.0f,
    -0.5f,  0.5f,  0.5f,   0.0f, 1.0f,   0.0f,  0.0f,  1.0f,
    // артқы жақ (z = -0.5) → (0, 0, -1)
     0.5f, -0.5f, -0.5f,   0.0f, 0.0f,   0.0f,  0.0f, -1.0f,
    -0.5f, -0.5f, -0.5f,   1.0f, 0.0f,   0.0f,  0.0f, -1.0f,
    -0.5f,  0.5f, -0.5f,   1.0f, 1.0f,   0.0f,  0.0f, -1.0f,
     0.5f, -0.5f, -0.5f,   0.0f, 0.0f,   0.0f,  0.0f, -1.0f,
    -0.5f,  0.5f, -0.5f,   1.0f, 1.0f,   0.0f,  0.0f, -1.0f,
     0.5f,  0.5f, -0.5f,   0.0f, 1.0f,   0.0f,  0.0f, -1.0f,
    // сол жақ (x = -0.5) → (-1, 0, 0)
    -0.5f, -0.5f, -0.5f,   0.0f, 0.0f,  -1.0f,  0.0f,  0.0f,
    -0.5f, -0.5f,  0.5f,   1.0f, 0.0f,  -1.0f,  0.0f,  0.0f,
    -0.5f,  0.5f,  0.5f,   1.0f, 1.0f,  -1.0f,  0.0f,  0.0f,
    -0.5f, -0.5f, -0.5f,   0.0f, 0.0f,  -1.0f,  0.0f,  0.0f,
    -0.5f,  0.5f,  0.5f,   1.0f, 1.0f,  -1.0f,  0.0f,  0.0f,
    -0.5f,  0.5f, -0.5f,   0.0f, 1.0f,  -1.0f,  0.0f,  0.0f,
    // оң жақ (x = +0.5) → (1, 0, 0)
     0.5f, -0.5f,  0.5f,   0.0f, 0.0f,   1.0f,  0.0f,  0.0f,
     0.5f, -0.5f, -0.5f,   1.0f, 0.0f,   1.0f,  0.0f,  0.0f,
     0.5f,  0.5f, -0.5f,   1.0f, 1.0f,   1.0f,  0.0f,  0.0f,
     0.5f, -0.5f,  0.5f,   0.0f, 0.0f,   1.0f,  0.0f,  0.0f,
     0.5f,  0.5f, -0.5f,   1.0f, 1.0f,   1.0f,  0.0f,  0.0f,
     0.5f,  0.5f,  0.5f,   0.0f, 1.0f,   1.0f,  0.0f,  0.0f,
    // асты (y = -0.5) → (0, -1, 0)
    -0.5f, -0.5f, -0.5f,   0.0f, 0.0f,   0.0f, -1.0f,  0.0f,
     0.5f, -0.5f, -0.5f,   1.0f, 0.0f,   0.0f, -1.0f,  0.0f,
     0.5f, -0.5f,  0.5f,   1.0f, 1.0f,   0.0f, -1.0f,  0.0f,
    -0.5f, -0.5f, -0.5f,   0.0f, 0.0f,   0.0f, -1.0f,  0.0f,
     0.5f, -0.5f,  0.5f,   1.0f, 1.0f,   0.0f, -1.0f,  0.0f,
    -0.5f, -0.5f,  0.5f,   0.0f, 1.0f,   0.0f, -1.0f,  0.0f,
    // үсті (y = +0.5) → (0, 1, 0)
    -0.5f,  0.5f,  0.5f,   0.0f, 0.0f,   0.0f,  1.0f,  0.0f,
     0.5f,  0.5f,  0.5f,   1.0f, 0.0f,   0.0f,  1.0f,  0.0f,
     0.5f,  0.5f, -0.5f,   1.0f, 1.0f,   0.0f,  1.0f,  0.0f,
    -0.5f,  0.5f,  0.5f,   0.0f, 0.0f,   0.0f,  1.0f,  0.0f,
     0.5f,  0.5f, -0.5f,   1.0f, 1.0f,   0.0f,  1.0f,  0.0f,
    -0.5f,  0.5f, -0.5f,   0.0f, 1.0f,   0.0f,  1.0f,  0.0f,
};

// Созылған куб y = -1.77-ге дейін түседі — еден сәл төменірек
const float FLOOR_Y   = -2.0f;
const int   GRID_HALF = 20;
const int   RING_N    = 12;
const float RING_R    = 10.0f;

std::vector<float> makeGrid() {
    std::vector<float> v;
    auto line = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c) {
        v.insert(v.end(), {a.x, a.y, a.z, c.r, c.g, c.b, b.x, b.y, b.z, c.r, c.g, c.b});
    };
    float h = (float)GRID_HALF;
    for (int i = -GRID_HALF; i <= GRID_HALF; i++) {
        float t = (float)i;
        glm::vec3 c = (i % 5 == 0) ? glm::vec3(0.45f) : glm::vec3(0.25f);
        line({t, FLOOR_Y, -h}, {t, FLOOR_Y, h}, i == 0 ? glm::vec3(0.2f, 0.4f, 1.0f) : c);
        line({-h, FLOOR_Y, t}, {h, FLOOR_Y, t}, i == 0 ? glm::vec3(1.0f, 0.3f, 0.3f) : c);
    }
    return v;
}

const float crosshair[] = {
    -0.03f,  0.0f,  0.0f,   1.0f, 1.0f, 1.0f,
     0.03f,  0.0f,  0.0f,   1.0f, 1.0f, 1.0f,
     0.0f,  -0.03f, 0.0f,   1.0f, 1.0f, 1.0f,
     0.0f,   0.03f, 0.0f,   1.0f, 1.0f, 1.0f,
};

GLuint makeColorVao(const float* data, size_t bytes, GLuint& vbo) {
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, bytes, data, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
    return vao;
}

// Сахна: 8-аптадағы бес куб. Біріншісі (0, 0, 0) — 3-тапсырмадағы созылған куб.
const glm::vec3 cubePositions[] = {
    glm::vec3( 0.0f,  0.0f,  0.0f),
    glm::vec3( 2.0f,  0.5f, -2.0f),
    glm::vec3(-1.8f, -0.3f, -1.0f),
    glm::vec3( 1.2f, -0.8f, -3.5f),
    glm::vec3(-2.2f,  1.0f, -4.0f),
};
const int CUBE_COUNT = sizeof(cubePositions) / sizeof(cubePositions[0]);

// 3-тапсырма: бірінші кубтың model матрицасы.
// GLM төменнен жоғары оқылады: алдымен 45° бұру (z осі), сосын y бойынша 2.5 есе созу —
// жақтары созу осіне қиғаш тұрады, қате тек осындайда көрінеді.
glm::mat4 firstCubeModel() {
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::scale (model, glm::vec3(1.0f, 2.5f, 1.0f));
    model = glm::rotate(model, glm::radians(45.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    return model;
}

// =====================================================================
//  --screenshots: README-дегі скриншоттар
// =====================================================================
struct ShotStep {
    const char* file;
    glm::vec3   camPos;
    glm::vec3   target;
    glm::vec3   light;
    float       time;                  // Қ3 түсі үшін
    void      (*setup)(LightState&);
};

const glm::vec3 ORIGIN(0.0f, 0.0f, 0.0f);
const glm::vec3 OVERVIEW(2.4f, 1.6f, 4.4f);         // алдыңғы + оң жақтар көрінеді
const glm::vec3 LEFT_VIEW(-3.0f, 1.0f, 3.4f);       // сол жақтан
const glm::vec3 FRONT_VIEW(0.0f, 0.0f, 5.0f);       // алдыңғы жаққа тура
const glm::vec3 RIGHT_VIEW(4.2f, 0.6f, 2.2f);       // қиғаш оң жақтар үлкен көрінеді
const glm::vec3 SIDE_LIGHT(2.3f, 0.0f, -0.4f);      // 3-тапсырма: жарық оң жақта

const ShotStep SHOTS[] = {
    {"01-basic.png",          OVERVIEW,  ORIGIN, START_LIGHT,               0, [](LightState&) {}},
    {"02-light-moving.png",   OVERVIEW,  ORIGIN, orbitPos(5.5f),            0, [](LightState& s) { s.orbit = true; }},
    {"03-no-max.png",         LEFT_VIEW, ORIGIN, glm::vec3(2.5f, 0.3f, 0.5f), 0, [](LightState& s) { s.noMax = true; s.ambient = 0.4f; }},
    {"03b-with-max.png",      LEFT_VIEW, ORIGIN, glm::vec3(2.5f, 0.3f, 0.5f), 0, [](LightState& s) { s.ambient = 0.4f; }},
    {"04-wrong-normal.png",   RIGHT_VIEW, ORIGIN, SIDE_LIGHT,               0, [](LightState& s) { s.wrongNormal = true; }},
    {"05-right-normal.png",   RIGHT_VIEW, ORIGIN, SIDE_LIGHT,               0, [](LightState&) {}},
    {"06-ambient-0.png",      OVERVIEW,  ORIGIN, glm::vec3(2.0f, 1.5f, -2.0f), 0, [](LightState& s) { s.ambient = 0.0f; }},
    {"06b-ambient-01.png",    OVERVIEW,  ORIGIN, glm::vec3(2.0f, 1.5f, -2.0f), 0, [](LightState& s) { s.ambient = 0.1f; }},
    {"07-ambient-05.png",     OVERVIEW,  ORIGIN, glm::vec3(2.0f, 1.5f, -2.0f), 0, [](LightState& s) { s.ambient = 0.5f; }},
    {"08-gouraud.png",        FRONT_VIEW, ORIGIN, FACE_LIGHT,               0, [](LightState& s) { s.gouraud = true; s.showLamp = false; }},
    {"09-phong.png",          FRONT_VIEW, ORIGIN, FACE_LIGHT,               0, [](LightState& s) { s.showLamp = false; }},
    {"10-directional.png",    glm::vec3(0.0f, 4.0f, 9.0f), glm::vec3(0.0f, 0.0f, -2.0f), START_LIGHT, 0,
                                                                              [](LightState& s) { s.directional = true; }},
    {"10b-point-far.png",     glm::vec3(0.0f, 4.0f, 9.0f), glm::vec3(0.0f, 0.0f, -2.0f), START_LIGHT, 0,
                                                                              [](LightState&) {}},
    {"11-color-light.png",    OVERVIEW,  ORIGIN, START_LIGHT,             4.0f, [](LightState& s) { s.colorLight = true; }},
    {"12-two-lights.png",     glm::vec3(0.0f, 1.2f, 5.0f), ORIGIN, glm::vec3(1.3f, 0.6f, 1.6f), 0,
                                                                              [](LightState& s) { s.twoLights = true; }},
};
const int SHOT_COUNT = sizeof(SHOTS) / sizeof(SHOTS[0]);

// =====================================================================
//  13-апта, 5-қадам: жарық uniform-дары. lightPos әр кадр өзгеруі мүмкін
//  (1-тапсырма), сондықтан бұл функция render циклінің ІШІНДЕ шақырылады.
// =====================================================================
void setLightUniforms(GLuint shader, const glm::vec3& color, const glm::vec3& color2) {
    glUseProgram(shader);
    glUniform3fv(glGetUniformLocation(shader, "lightPos"),   1, glm::value_ptr(lightPos));
    glUniform3fv(glGetUniformLocation(shader, "lightColor"), 1, glm::value_ptr(color));
    glUniform1f (glGetUniformLocation(shader, "ambientStrength"), ls.ambient);
    glUniform1i (glGetUniformLocation(shader, "useMax"),      !ls.noMax);
    glUniform1i (glGetUniformLocation(shader, "wrongNormal"), ls.wrongNormal);
    glUniform1i (glGetUniformLocation(shader, "directional"), ls.directional);
    glUniform3fv(glGetUniformLocation(shader, "lightDirection"), 1, glm::value_ptr(SUN_DIRECTION));
    glUniform1i (glGetUniformLocation(shader, "twoLights"),   ls.twoLights);
    glm::vec3 pos2 = secondLightPos(lightPos);
    glUniform3fv(glGetUniformLocation(shader, "lightPos2"),   1, glm::value_ptr(pos2));
    glUniform3fv(glGetUniformLocation(shader, "lightColor2"), 1, glm::value_ptr(color2));
}

// =====================================================================
//  MAIN
// =====================================================================
int main(int argc, char** argv) {
    bool screenshotMode = (argc > 1 && std::strcmp(argv[1], "--screenshots") == 0);

    if (!glfwInit()) {
        std::cerr << "GLFW іске қосылмады\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(gWidth, gHeight,
                                          "Компьютерлік графика — 13-апта",
                                          nullptr, nullptr);
    if (!window) {
        std::cerr << "Терезе жасалмады. Видеокарта OpenGL 3.3-ті қолдамауы мүмкін.\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSwapInterval(1);

    if (gladLoadGL(glfwGetProcAddress) == 0) {
        std::cerr << "GLAD жүктелмеді\n";
        glfwTerminate();
        return -1;
    }

    applyMouseMode(window);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    if (glfwRawMouseMotionSupported())
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

    glfwGetFramebufferSize(window, &gWidth, &gHeight);
    glViewport(0, 0, gWidth, gHeight);

    std::cout << "OpenGL: " << glGetString(GL_VERSION) << "\n";
    std::cout << "1 — орбита, 2 — max жоқ, 3 — mat3(model), O/L — ambient, 5 — Гуро,\n"
                 "H — шам, 6 — бағытталған, 7 — түрлі-түсті, 8 — екі көз,\n"
                 "← → ↑ ↓ PgUp PgDn — жарықты жылжыту, C — жақтың ортасына,\n"
                 "P — күй, R — бастапқы күй, M — тінтуір режимі, F12 — скриншот\n";
    if (dragMode)
        std::cout << "WSL: сүйреу режимі — камераны бұру үшін ЛКМ басып тұрып тартыңыз (M — ауыстыру)\n";

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    GLuint phongShader   = createShaderProgram(
        phongVertexSrc,
        std::string(phongFragmentHead) + lightingGlsl + phongFragmentMain);
    GLuint gouraudShader = createShaderProgram(
        std::string(gouraudVertexHead) + lightingGlsl + gouraudVertexMain,
        gouraudFragmentSrc);
    GLuint lampShader    = createShaderProgram(lampVertexSrc, lampFragmentSrc);
    GLuint colorShader   = createShaderProgram(colorVertexSrc, colorFragmentSrc);

    GLint cModel = glGetUniformLocation(colorShader, "model");
    GLint cView  = glGetUniformLocation(colorShader, "view");
    GLint cProj  = glGetUniformLocation(colorShader, "projection");

    // --- 13-апта, 2-қадам: поз + UV + нормаль, stride ҮШ жерде де 8 ---
    GLuint cubeVao, cubeVbo;
    glGenVertexArrays(1, &cubeVao);
    glGenBuffers(1, &cubeVbo);
    glBindVertexArray(cubeVao);
    glBindBuffer(GL_ARRAY_BUFFER, cubeVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
    // поз: location 0
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // UV: location 1
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                          (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // нормаль: location 2
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                          (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);

    GLuint gridVbo, crossVbo;
    std::vector<float> grid = makeGrid();
    int    gridVerts = (int)(grid.size() / 6);
    GLuint gridVao   = makeColorVao(grid.data(), grid.size() * sizeof(float), gridVbo);
    GLuint crossVao  = makeColorVao(crosshair, sizeof(crosshair), crossVbo);

    // --- Текстура: 11-апта, бір ғана container.png, 0-бірлік ---
    std::string texPath = std::string(ASSET_DIR) + "/container.png";
    unsigned int texture = loadTexture(texPath.c_str());
    for (GLuint prog : { phongShader, gouraudShader }) {
        glUseProgram(prog);
        glUniform1i(glGetUniformLocation(prog, "ourTexture"), 0);
    }

    double lastFrame  = glfwGetTime();
    double lastTitle  = 0.0;
    int    frame      = 0;
    int    shotIndex  = 0;
    int    shotLoaded = -1;
    float  colorTime  = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        double currentFrame = glfwGetTime();
        float  dt = (float)(currentFrame - lastFrame);
        lastFrame = currentFrame;

        // ------------------------- ЖАҢАРТУ ---------------------------
        if (screenshotMode) {
            const ShotStep& s = SHOTS[shotIndex];
            if (shotLoaded != shotIndex) {
                ls = LightState();
                s.setup(ls);
                shotLoaded = shotIndex;
            }
            angle     = 0.6f;
            lightPos  = s.light;
            colorTime = s.time;
            fov       = 45.0f;
            aimCamera(s.camPos, s.target);
        } else {
            angle += dt;
            processInput(window, dt);
            colorTime = (float)glfwGetTime();
            // 1-тапсырма: жарық көзі шеңбер бойымен
            if (ls.orbit)
                lightPos = orbitPos((float)glfwGetTime());
        }

        // Жарықтың түсі: ақ; Қ3 — уақыт бойынша; Қ4 — қызыл мен көк
        glm::vec3 lightColor  = ls.colorLight ? rainbowColor(colorTime)
                              : ls.twoLights  ? RED : WHITE;
        glm::vec3 lightColor2 = BLUE;

        glm::mat4 view = glm::lookAt(camPos, camPos + camFront, camUp);
        glm::mat4 projection = glm::perspective(glm::radians(fov),
            (float)gWidth / (float)gHeight, 0.1f, 100.0f);

        if (currentFrame - lastTitle > 0.25) {   // WSLg-де тақырыпты әр кадрда жаңартпаймыз
            lastTitle = currentFrame;
            char buf[256];
            std::snprintf(buf, sizeof(buf), "%s | ambient %.2f%s%s%s%s%s%s",
                          ls.gouraud ? "Гуро" : "Фонг", ls.ambient,
                          ls.orbit ? " | орбита" : "",
                          ls.noMax ? " | max ЖОҚ" : "",
                          ls.wrongNormal ? " | mat3(model) ҚАТЕ" : "",
                          ls.directional ? " | бағытталған" : "",
                          ls.colorLight ? " | түрлі-түсті" : "",
                          ls.twoLights ? " | екі көз" : "");
            glfwSetWindowTitle(window, buf);
        }

        // --------------------------- СЫЗУ ----------------------------
        glClearColor(0.10f, 0.12f, 0.18f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        GLuint shader = ls.gouraud ? gouraudShader : phongShader;
        setLightUniforms(shader, lightColor, lightColor2);   // glUseProgram ішінде
        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE,
                           glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE,
                           glm::value_ptr(view));
        GLint uModel  = glGetUniformLocation(shader, "model");
        GLint uNormal = glGetUniformLocation(shader, "normalMatrix");

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glBindVertexArray(cubeVao);

        auto drawCube = [&](const glm::mat4& model) {
            glUniformMatrix4fv(uModel, 1, GL_FALSE, glm::value_ptr(model));
            // 5-қадам: normalMatrix ӘР КУБ үшін бөлек — әрқайсысының өз model-і бар.
            // mat3, mat4 емес: нормальға жылжыту керек емес.
            glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(model)));
            glUniformMatrix3fv(uNormal, 1, GL_FALSE, glm::value_ptr(normalMatrix));
            glDrawArrays(GL_TRIANGLES, 0, 36);
        };

        for (int i = 0; i < CUBE_COUNT; i++) {
            glm::mat4 model;
            if (i == 0) {
                model = firstCubeModel();                       // 3-тапсырма
            } else {
                model = glm::translate(glm::mat4(1.0f), cubePositions[i]);
                model = glm::rotate(model, angle * (0.5f + i * 0.3f), glm::vec3(0.3f, 1.0f, 0.0f));
            }
            drawCube(model);
        }
        for (int i = 0; i < RING_N; i++) {
            float a = glm::radians(360.0f / RING_N * i);
            glm::vec3 p(RING_R * cos(a), FLOOR_Y + 0.5f + (i % 3), RING_R * sin(a));
            glm::mat4 model = glm::translate(glm::mat4(1.0f), p);
            model = glm::rotate(model, angle * 0.4f, glm::vec3(0.0f, 1.0f, 0.0f));
            drawCube(model);
        }

        // --- Қ1: жарық көзінің орнында кішкентай куб (сол VAO, басқа model) ---
        if (ls.showLamp && !ls.directional) {
            glUseProgram(lampShader);
            glUniformMatrix4fv(glGetUniformLocation(lampShader, "projection"), 1, GL_FALSE,
                               glm::value_ptr(projection));
            glUniformMatrix4fv(glGetUniformLocation(lampShader, "view"), 1, GL_FALSE,
                               glm::value_ptr(view));
            auto drawLamp = [&](const glm::vec3& pos, const glm::vec3& color) {
                glm::mat4 lampModel = glm::translate(glm::mat4(1.0f), pos);
                lampModel = glm::scale(lampModel, glm::vec3(0.1f));
                glUniformMatrix4fv(glGetUniformLocation(lampShader, "model"), 1, GL_FALSE,
                                   glm::value_ptr(lampModel));
                glUniform3fv(glGetUniformLocation(lampShader, "lampColor"), 1,
                             glm::value_ptr(color));
                glDrawArrays(GL_TRIANGLES, 0, 36);
            };
            drawLamp(lightPos, lightColor);
            if (ls.twoLights) drawLamp(secondLightPos(lightPos), lightColor2);
        }

        // Еден-торы мен прицел — түсті шейдермен
        glUseProgram(colorShader);
        glm::mat4 identity(1.0f);
        glUniformMatrix4fv(cProj, 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(cView, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(cModel, 1, GL_FALSE, glm::value_ptr(identity));
        glBindVertexArray(gridVao);
        glDrawArrays(GL_LINES, 0, gridVerts);

        if (!screenshotMode) {
            float aspect = (float)gWidth / (float)gHeight;
            glm::mat4 crossModel = glm::scale(identity, glm::vec3(1.0f / aspect, 1.0f, 1.0f));
            glUniformMatrix4fv(cModel, 1, GL_FALSE, glm::value_ptr(crossModel));
            glUniformMatrix4fv(cView,  1, GL_FALSE, glm::value_ptr(identity));
            glUniformMatrix4fv(cProj,  1, GL_FALSE, glm::value_ptr(identity));
            glDisable(GL_DEPTH_TEST);
            glBindVertexArray(crossVao);
            glDrawArrays(GL_LINES, 0, 4);
            glEnable(GL_DEPTH_TEST);
        }

        if (screenshotMode && ++frame % 10 == 0) {
            saveScreenshot(std::string(SCREENSHOT_DIR) + "/" + SHOTS[shotIndex].file);
            if (++shotIndex == SHOT_COUNT) glfwSetWindowShouldClose(window, true);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &cubeVao);
    glDeleteVertexArrays(1, &gridVao);
    glDeleteVertexArrays(1, &crossVao);
    glDeleteBuffers(1, &cubeVbo);
    glDeleteBuffers(1, &gridVbo);
    glDeleteBuffers(1, &crossVbo);
    glDeleteTextures(1, &texture);
    glDeleteProgram(phongShader);
    glDeleteProgram(gouraudShader);
    glDeleteProgram(lampShader);
    glDeleteProgram(colorShader);

    glfwTerminate();
    return 0;
}
