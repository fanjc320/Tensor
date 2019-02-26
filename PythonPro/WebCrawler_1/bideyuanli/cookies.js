function switchFeature( name ) {
    var feature = getCookie( '1and1-wp-assistant-config-' + name );

    if ( ! feature || feature === '0' ) {
        setCookie( '1and1-wp-assistant-config-' + name, 1 );
    } else {
        setCookie( '1and1-wp-assistant-config-' + name, 0 );
    }
    window.location.reload();
}

function setCookie( param, value, exdays ) {
    var date = new Date();
    date.setTime( date.getTime() + ( exdays * 24 * 60 * 60 * 1000 ) );

    var expires = 'expires=' + date.toUTCString();

    document.cookie = param + '=' + value + ';' + expires + ';path=/';
}

function getCookie( param ) {
    var name = param + '=';
    var cookie_list = document.cookie.split(';');

    for ( var i = 0; i < cookie_list.length; i++ ) {
        var cookie = cookie_list[i];

        while ( cookie.charAt( 0 ) === ' ' ) {
            cookie = cookie.substring( 1 );
        }
        if ( cookie.indexOf( name ) === 0 ) {
            return cookie.substring( name.length, cookie.length );
        }
    }
    return '';
}