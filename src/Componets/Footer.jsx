import React from "react";
import { Link } from "react-router-dom";
import PhoneInput from "react-phone-input-2";
import "react-phone-input-2/lib/style.css";

const Footer = () => {
  return (
    <div className="z-50 shadow-md fixed w-full bottom-0 bg-white border-t border-black">
      <ul className="flex flex-row justify-around p-[10px] bg-[#6e5e31] md:hidden">
        <li>
          <Link className="text-[19px] text-white">
            <span className=" border-b-white">Enquire Now</span>
          </Link>
        </li>
        <li>
          <Link className="text-[19px] text-white border-r border-white"></Link>
        </li>
        <li>
          <Link className="text-[19px] text-white p-[5px]">
            <span className="border-b-white">Call Now</span>
          </Link>
        </li>
      </ul>

      <footer className="hidden md:block">
        <div className="lg:flex lg:flex-row lg:justify-around lg:pt-3 lg:pb-2">
          <div className="flex items-center justify-center">
            <h2 className="text-base md:text-lg lg:text-xl text-black">
              Enquire Now
            </h2>
          </div>
          <div className="">
            <form
              className="form grid grid-cols-2 lg:grid-cols-4 lg:justify-around  w-full"
              id="global_form"
              method="POST"
            >
              <div className="border border-black  ml-5 mr-5  rounded">
                <input
                  type="text"
                  className="h-full w-full"
                  id="name"
                  name="name"
                  placeholder="Name"
                  required=""
                />
              </div>
              <div className="border border-black mr-5 h-[39px] rounded">
                <PhoneInput
                  country={"in"}
                  placeholder="Mobile"
                  className="h-full w-full"
                />
              </div>
              <div className="border border-black ml-5 mr-5 lg:ml-0 h-[39px] rounded">
                <input
                  type="email"
                  id="email"
                  name="email"
                  placeholder="Email"
                  className="h-full w-full"
                />
              </div>
              <div className="flex justify-center">
                <button
                  type="submit"
                  className="border border-black mr-5 h-[39px] rounded p-2 font-extrabold text-white bg-[#6e5e31]"
                >
                  SUBMIT
                </button>
              </div>
            </form>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Footer;
