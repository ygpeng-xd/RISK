from Crypto.Cipher import AES
import hashlib
import base64


# 密钥（key）, 密斯偏移量（iv） CBC模式加密
BLOCK_SIZE = 16  # Bytes
pad = lambda s: s + (BLOCK_SIZE - len(s) % BLOCK_SIZE) * chr(BLOCK_SIZE - len(s) % BLOCK_SIZE)
unpad = lambda s: s[:-ord(s[len(s) - 1:])]
vi = '0102030405060708'


def AES_Encrypt(key, data):
    data = pad(data)
    # 字符串补位
    cipher = AES.new(key.encode('utf8'), AES.MODE_CBC, vi.encode('utf8'))
    encryptedbytes = cipher.encrypt(data.encode('utf8'))
    # 加密后得到的是bytes类型的数据，使用Base64进行编码,返回byte字符串
    encodestrs = base64.b64encode(encryptedbytes)
    # 对byte字符串按utf-8进行解码
    enctext = encodestrs.decode('utf8')
    return enctext


def AES_Decrypt(key, data):
    data = data.encode('utf8')
    encodebytes = base64.decodebytes(data)
    # 将加密数据转换位bytes类型数据
    cipher = AES.new(key.encode('utf8'), AES.MODE_CBC, vi.encode('utf8'))
    text_decrypted = cipher.decrypt(encodebytes)
    # 去补位
    text_decrypted = unpad(text_decrypted)
    text_decrypted = text_decrypted.decode('utf8')
    return text_decrypted


def sha1_Encrypt(data):
    sha1 = hashlib.sha1()
    sha1.update(data.encode('utf-8'))
    ciphertext = sha1.hexdigest()
    return ciphertext


if __name__ == '__main__':
    # AES CBC模式加解密
    key = 'xducc02241931xdu'
    aes_data = '12345678876543211234567887654321123456788765432'

    aes_ciphertext = AES_Encrypt(key, aes_data)
    print(aes_ciphertext)    # type(aes_ciphertext) = str

    plaintext = AES_Decrypt(key, aes_ciphertext)
    print(plaintext)           # type(plaintext) = str
    print()

    # sha1加密
    sha1_data = 'indirect 00000'
    sha1_ciphertext = sha1_Encrypt(sha1_data)
    print(sha1_ciphertext)      # type(sha1_ciphertext) = str
