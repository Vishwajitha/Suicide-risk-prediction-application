-- Create a master key (only once per DB)
CREATE MASTER KEY ENCRYPTION BY PASSWORD = 'vishwa25';

-- Create a certificate for encryption
CREATE CERTIFICATE AppConfigCert
WITH SUBJECT = 'App Config Certificate';

-- Create a symmetric key using the certificate
CREATE SYMMETRIC KEY AppConfigKey
WITH ALGORITHM = AES_256
ENCRYPTION BY CERTIFICATE AppConfigCert;

-- Drop if already exists, then recreate
DROP TABLE IF EXISTS AppConfig;

CREATE TABLE AppConfig (
    id INT PRIMARY KEY IDENTITY(1,1),
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value VARBINARY(MAX) NOT NULL
);

OPEN SYMMETRIC KEY AppConfigKey DECRYPTION BY CERTIFICATE AppConfigCert;

INSERT INTO AppConfig (config_key, config_value)
VALUES ('api_key', EncryptByKey(Key_GUID('AppConfigKey'), 'k4aAgrvgf51StMMCMav6a4BrQ'));

INSERT INTO AppConfig (config_key, config_value)
VALUES ('api_secret', EncryptByKey(Key_GUID('AppConfigKey'), 'JMIuEjbSUcFCg89NSiRdEhN2F5nBUAKFGsUdo9r91zGqQhEMYX'));

INSERT INTO AppConfig (config_key, config_value)
VALUES ('bearer_token', EncryptByKey(Key_GUID('AppConfigKey'), 'AAAAAAAAAAAAAAAAAAAAAFpevQEAAAAAz4AV0u/nJxoJ+Xje7f+yrN8pSLY=b5Y35gvUO9U2MBDzt8Hwbony29OdkKZE6xAYtHNoig1WsoVCSy'));

INSERT INTO AppConfig (config_key, config_value)
VALUES ('access_token', EncryptByKey(Key_GUID('AppConfigKey'), '1821233002464280576-QBbt1LW94GooGqcrrPdKLepOQRnSGS'));

INSERT INTO AppConfig (config_key, config_value)
VALUES ('access_token_secret', EncryptByKey(Key_GUID('AppConfigKey'), 'RwephwK46qgJMerRMLkFDiRiJCw44eSTkqcHZWJFgcGgA'));

INSERT INTO AppConfig (config_key, config_value)
VALUES ('suicide_message', EncryptByKey(Key_GUID('AppConfigKey'), 'Hello! If you’re feeling like you need support, please reach out to us. You are not alone.'));

CLOSE SYMMETRIC KEY AppConfigKey;


CREATE PROCEDURE sp_GetDecryptedConfig
    @key VARCHAR(255)
AS
BEGIN
    OPEN SYMMETRIC KEY AppConfigKey DECRYPTION BY CERTIFICATE AppConfigCert;

    SELECT 
        config_key,
        CONVERT(VARCHAR(MAX), DecryptByKey(config_value)) AS config_value
    FROM AppConfig
    WHERE config_key = @key;

    CLOSE SYMMETRIC KEY AppConfigKey;
END

select * from AppConfig
