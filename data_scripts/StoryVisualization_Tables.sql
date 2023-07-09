CREATE TABLE Stories (
    scene_id VARCHAR(255),
    globalID VARCHAR(255) PRIMARY KEY,
    scene_id_1 VARCHAR(255),
    scene_id_2 VARCHAR(255),
    scene_id_3 VARCHAR(255),
    scene_id_4 VARCHAR(255),
    scene_id_5 VARCHAR(255),
    description TEXT,
    setting TEXT
);

CREATE TABLE Scenes (
    globalID VARCHAR(255) PRIMARY KEY,
    description TEXT,
    setting TEXT
);

CREATE TABLE Characters (
    id INT AUTO_INCREMENT,
    scene_id VARCHAR(255),
    globalID VARCHAR(255),
    entityLabel VARCHAR(255),
    entitySpan VARCHAR(255),
    labelNPC VARCHAR(255),
    PRIMARY KEY(id),
    FOREIGN KEY (scene_id) REFERENCES Scenes(globalID)
);

CREATE TABLE Actions (
    id INT AUTO_INCREMENT,
    character_id INT,
    action VARCHAR(255),
    PRIMARY KEY(id),
    FOREIGN KEY (character_id) REFERENCES Characters(id)
);

CREATE TABLE Rectangles (
    id INT AUTO_INCREMENT,
    character_id INT,
    rectangle VARCHAR(255),
    PRIMARY KEY(id),
    FOREIGN KEY (character_id) REFERENCES Characters(id)
);

CREATE TABLE KeyFrames (
    id INT AUTO_INCREMENT,
    scene_id VARCHAR(255),
    keyFrame VARCHAR(255),
    PRIMARY KEY(id),
    FOREIGN KEY (scene_id) REFERENCES Scenes(globalID)
);

CREATE TABLE Objects (
    id INT AUTO_INCREMENT,
    scene_id VARCHAR(255),
    globalID VARCHAR(255),
    entityLabel VARCHAR(255),
    entitySpan VARCHAR(255),
    labelNPC VARCHAR(255),
    PRIMARY KEY(id),
    FOREIGN KEY (scene_id) REFERENCES Scenes(globalID)
);

CREATE TABLE ObjectRectangles (
    id INT AUTO_INCREMENT,
    object_id INT,
    rectangle VARCHAR(255),
    PRIMARY KEY(id),
    FOREIGN KEY (object_id) REFERENCES Objects(id)
);

-- CREATE TABLE CorefClusters (
--     id INT AUTO_INCREMENT,
--     scene_id VARCHAR(255),
--     cluster VARCHAR(255),
--     PRIMARY KEY(id),
--     FOREIGN KEY (scene_id) REFERENCES Scenes(globalID)
-- );

-- CREATE TABLE PosTags (
--     id INT AUTO_INCREMENT,
--     scene_id VARCHAR(255),
--     posTag VARCHAR(255),
--     PRIMARY KEY(id),
--     FOREIGN KEY (scene_id) REFERENCES Scenes(globalID)
-- );
