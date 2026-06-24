
function anonId(r) {
    var match = (r.headersIn.Cookie || '').match(/(?:^|;\s*)aid=([^;]+)/);
    if (match && match[1]) {
        return match[1];
    }
    if (!r.kingChatAid) {
        var id = 'anon-' + Math.random().toString(36).slice(2, 14) +
                 Math.round(Math.random() * 1e9).toString(36);
        r.kingChatAid = id;
        r.headersOut['Set-Cookie'] =
            'aid=' + id + '; Path=/; Max-Age=31536000; HttpOnly; Secure; SameSite=Lax';
    }
    return r.kingChatAid;
}

export default { anonId };
