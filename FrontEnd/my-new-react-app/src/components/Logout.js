import React from 'react';
// import { GoogleLogout } from 'react-google-login';
import { googleLogout } from '@react-oauth/google';

function Logout({ setUser, clientId }) {
  // const onSuccess = () => {
  //   googleLogout();
  //   setUser(null);
  //   localStorage.setItem("login", null);
  //   console.log('Logout made successfully');
  // };

  const logOut = () => {
    googleLogout();
    setUser(null);
    localStorage.setItem("login", null);
    console.log('Logout made successfully');
  };

  return (
    <div>
      <button onClick={logOut}
              type="button"
              class="btn btn-light">Log out
      </button>
      {/* <GoogleLogout
        clientId={clientId}
        buttonText="Logout"
        onLogoutSuccess={onSuccess}
      ></GoogleLogout> */}
    </div>
  );
}

export default Logout;