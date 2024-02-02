import { Link } from "react-scroll";

export const Header = () => {
  function handleClick() {
    if (collapseMenu.style.display === "block") {
      collapseMenu.style.display = "none";
    } else {
      collapseMenu.style.display = "block";
    }
  }

  return (
    <header className="shadow-md bg-white fixed top-0 w-full z-10 ">
      <section className="flex border border-black flex-col md:flex-row md:justify-center md:items-center md:max-lg:gap-7 lg:justify-around justify-between">
        <div className="flex justify-between ">
          <img
            src="./logo.jpg"
            alt="ISL Logo"
            className="max-w-[21vw] h-[85px]"
          />

          <div className="flex items-center px-10 md:order-1 overflow-x-auto md:hidden">
            <button
              id="toggle"
              className=" rounded p-2 bg-[#6e5e31]"
              onClick={handleClick}
            >
              <svg
                className="w-8 h-6 text-white "
                fill="#FFFFFF"
                viewBox="0 0 20 20"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  fill-rule="evenodd"
                  d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z"
                  clip-rule="evenodd"
                ></path>
              </svg>
            </button>
          </div>
        </div>

        <div>
          <ul
            id="collapseMenu"
            className="font-[Philosopher-Regular] md:!flex justify-center block  text-center w-full md:gap-6 "
          >
            <li className=" py-2 cursor-pointer">
              <Link
                to="AboutID"
                smooth
                duration={500}
                className="text-[16px] block"
              >
                Overview
              </Link>
            </li>
            <li className="  py-2 cursor-pointer">
              <Link
                to="ImageSliderID"
                smooth
                duration={500}
                className=" text-[16px] block"
              >
                Gallery
              </Link>
            </li>
            <li className=" py-2 cursor-pointer">
              <Link
                to="FloorPlanID"
                smooth
                duration={500}
                className=" text-[16px] block"
              >
                Floor Plan
              </Link>
            </li>
            <li className="  py-2 cursor-pointer">
              <Link
                to="LocationID"
                smooth
                duration={500}
                className=" text-[16px] block"
              >
                Location
              </Link>
            </li>
            <li className="  py-2 cursor-pointer">
              <Link
                to="AboutDeveloperID"
                smooth
                duration={500}
                className="text-[16px] block"
              >
                About
              </Link>
            </li>
            <li className="flex items-center justify-center gap-2 cursor-pointer ">
              <img src="./phone-call.png" alt="ISL Logo" className="h-[20px]" />
              <Link to="/" className=" text-[16px] block">
                +912241498607
              </Link>
            </li>
          </ul>
        </div>
      </section>
    </header>
  );
};
