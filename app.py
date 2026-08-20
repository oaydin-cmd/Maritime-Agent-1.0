def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 'messages' tablosu var mı kontrol et
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
    table_exists = cursor.fetchone()
    
    if table_exists:
        # Tablodaki mevcut sütunları al
        cursor.execute("PRAGMA table_info(messages)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Eğer yeni eklenen sütunlar eski veritabanında yoksa tabloyu silip yenisini oluştur
        if "excel_blob" not in columns:
            cursor.execute("DROP TABLE messages")
            conn.commit()

    # Güncel şema ile tabloyu baştan oluştur
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            role TEXT,
            content TEXT,
            docx_blob BLOB,
            docx_file_name TEXT,
            excel_blob BLOB,
            excel_file_name TEXT
        )
    ''')
    conn.commit()
    conn.close()
