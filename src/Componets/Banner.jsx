import React from "react";

const Banner = () => {
  const backgroundImageUrl1 = "./banner.png";
  const backgroundImageUrl2 = "./bannersm.png";
  return (
    <>
      <div
        className="object-cover relative hidden sm:block"
        style={{
          backgroundImage: `url(${backgroundImageUrl1})`,
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat",
          height: "634px",
          marginTop: "85px",
          zIndex: 0,
        }}
      ></div>
      <div
        className=" relative sm:hidden"
        style={{
          backgroundImage: `url(${backgroundImageUrl2})`,
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat",
          height: "634px",
          marginTop: "85px",
          zIndex: 0,
        }}
      ></div>
    </>
  );
};

export default Banner;
