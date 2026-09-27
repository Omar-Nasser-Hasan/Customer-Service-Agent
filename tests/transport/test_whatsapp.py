import hashlib
import hmac

from customer_service.config.settings import Settings
from customer_service.transport.whatsapp import WhatsAppTransport


def settings():
    return Settings(whatsapp_phone_encryption_key="1H1QzAJUytPJswyikOd96HTOkxRC05tixSYag2wSWqI=", whatsapp_thread_hmac_key="thread-secret")


def test_signature_thread_identifier_and_encrypted_phone():
    body = b'{"object":"whatsapp_business_account"}'
    signature = "sha256=" + hmac.new(b"app-secret", body, hashlib.sha256).hexdigest()
    assert WhatsAppTransport.valid_signature(body, signature, "app-secret")
    assert not WhatsAppTransport.valid_signature(body, signature, "other")
    transport = WhatsAppTransport(settings())
    assert transport.thread_id("20123456789") != "20123456789"
    encrypted = transport.encrypt_phone("20123456789")
    assert "20123456789" not in encrypted
    assert transport.decrypt_phone(encrypted) == "20123456789"
