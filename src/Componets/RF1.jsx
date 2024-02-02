import React from "react";
import PhoneInput from "react-phone-input-2";

const RF1 = ({ isOpen, onClose }) => {
  return (
    <div
      className={`fixed inset-0 ${
        isOpen ? "flex" : "hidden"
      } items-center justify-center z-10 bg-black  bg-opacity-50`}
    >
      <div className="absolute bg-white p-1 pb-3 rounded-md shadow-lg max-w-md w-[406px] text-center">
        <div className="flex float-end mb-[8px]">
          <span onClick={onClose} className=" bg-white cursor-pointer ">
            <b>X</b>
          </span>
        </div>
        <div className="mt-[20px]">
          <h1 className="text-3xl mb-4 ">Enquire Now</h1>
          <p className=" mb-4">Register Here And Avail The Best Offers!!</p>
        </div>
        <div>
          <form
            className="flex flex-col gap-2 justify-center w-full"
            id="global_form"
            method="POST"
          >
            <div className="border border-black h-[34px]  ml-5 mr-5 ">
              <input
                type="text"
                className="h-full w-full"
                id="name"
                name="name"
                placeholder="Name"
                required=""
              />
            </div>
            <div className="border border-black ml-5 mr-5">
              <PhoneInput
                country={"in"}
                placeholder="Mobile"
                // className="h-full w-full"
              />
            </div>
            <div className="border border-black ml-5 mr-5  h-[34px]">
              <input
                type="email"
                id="email"
                name="email"
                placeholder="Email"
                className="h-full w-full"
              />
            </div>
            <div className="flex justify-center ml-5 mr-5 gap-1">
              <input
                type="checkbox"
                value="agree"
                id="dnd"
                name="agree"
                required=""
                checked=""
              />
              <label className="text-[8px]" for="dnd" class="input_label">
                I agree and authorize team to contact me. This will override the
                registry with DNC / NDNC
              </label>
            </div>
            <div className="flex justify-center">
              <button
                type="submit"
                className="border border-black mr-5 h-[39px] rounded p-2 font-bold text-white bg-[#6e5e31]"
              >
                SUBMIT
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default RF1;
