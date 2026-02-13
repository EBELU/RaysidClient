class BytesBuffer:
    def __init__(self):
        self.buf = bytearray()
        
    def feed(self, data: bytes):
        self.buf.extend(data)
        
    def unpack(self):
        expected_len = int(self.buf[0])
        packet_type = self.buf[1]
        
        if packet_type not in (0x01, 0x17, 0x30, 0x31, 0x32):
            self.buf.clear()
        
        if len(self.buf) >= expected_len:
            packet = bytes(self.buf[:expected_len ])
            del self.buf[:expected_len]