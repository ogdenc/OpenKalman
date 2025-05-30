/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Tests for \ref values::complex types.
 */

#include <type_traits>
#include "values/tests/tests.hpp"
#include "values/concepts/number.hpp"
#include "values/concepts/complex.hpp"
#include "values/concepts/integral.hpp"
#include "values/concepts/floating.hpp"
#include "values/concepts/dynamic.hpp"

using namespace OpenKalman;


#if defined(__GNUC__) or defined(__clang__)
#define COMPLEXINTEXISTS(F) F
#else
#define COMPLEXINTEXISTS(F)
#endif

#include "values/traits/number_type_of_t.hpp"
#include "values/traits/real_type_of_t.hpp"

TEST(values, interface)
{
  static_assert(std::is_same_v<values::number_type_of_t<std::complex<double>>, std::complex<double>>);
  static_assert(std::is_same_v<values::real_type_of_t<std::complex<double>>, double>);
  static_assert(std::is_same_v<values::real_type_of_t<double>, double>);
  static_assert(std::is_same_v<values::real_type_of_t<int>, double>);
}


TEST(values, complex)
{
  static_assert(values::number<std::complex<double>>);
  static_assert(values::number<std::complex<float>>);
  COMPLEXINTEXISTS(static_assert(values::number<std::complex<int>>));

  static_assert(values::complex<std::complex<double>>);
  static_assert(values::complex<std::complex<float>>);
  static_assert(not values::complex<double>);
  COMPLEXINTEXISTS(static_assert(values::complex<std::complex<int>>));

  COMPLEXINTEXISTS(static_assert(not values::integral<std::complex<int>>));
  static_assert(not values::floating<std::complex<float>>);
  static_assert(not values::floating<std::complex<double>>);

  static_assert(values::dynamic<std::complex<double>>);
  static_assert(values::dynamic<std::complex<float>>);
}

#include "values/concepts/not_complex.hpp"
#include "values/classes/Fixed.hpp"

TEST(values, Fixed_complex)
{
  static_assert(values::not_complex<int>);
  static_assert(values::not_complex<double>);
  static_assert(not values::not_complex<std::complex<double>>);

  static_assert(values::complex<values::Fixed<std::complex<double>, 3, 0>>);

  static_assert(values::not_complex<values::Fixed<std::complex<double>, 3, 0>>);

  static_assert(values::not_complex<values::Fixed<std::complex<double>, 3, 0>>);
  static_assert(not values::not_complex<values::Fixed<std::complex<double>, 3, 1>>);

  static_assert(std::real(values::to_number(values::Fixed<std::complex<double>, 3, 4>{})) == 3);
  static_assert(std::imag(values::to_number(values::Fixed<std::complex<double>, 3, 4>{})) == 4);
  static_assert(values::Fixed<std::complex<double>, 3, 4>{}() == std::complex<double>{3, 4});

  EXPECT_TRUE(test::is_near(values::Fixed<std::complex<double>, 3, 4>{}, values::Fixed<std::complex<double>, 4, 3>{}, 2));
  EXPECT_TRUE(test::is_near(values::Fixed<std::complex<double>, 3, 4>{}, std::complex<double>{2, 5}, std::complex<double>{2, 2}));
  EXPECT_TRUE(test::is_near(std::complex<double>{3 - 1e-9, 4 + 1e-9}, values::Fixed<std::complex<double>, 3, 4>{}, 1e-6));
  EXPECT_TRUE(test::is_near(values::Fixed<std::complex<double>, 3, 0>{}, 3. - 1e-9, 1e-6));
}


#include "values/functions/internal/make_complex_number.hpp"

TEST(values, make_complex_number)
{
  static_assert(values::internal::make_complex_number<std::complex<double>>(3., 4.) == std::complex<double>{3, 4});
  static_assert(values::internal::make_complex_number<std::complex<double>>(3., 4.f) == std::complex<double>{3, 4});
  static_assert(values::internal::make_complex_number<std::complex<double>>(std::complex<float>{3, 4}) == std::complex<double>{3, 4});
  static_assert(values::internal::make_complex_number<double>(std::complex<double>{3, 4}) == std::complex<double>{3, 4});
  static_assert(values::internal::make_complex_number<double>(std::complex<float>{3, 4}) == std::complex<double>{3, 4});
  static_assert(values::internal::make_complex_number(3., 4.) == std::complex<double>{3, 4});
  static_assert(values::internal::make_complex_number(3., 4.f) == std::complex<double>{3, 4});
  static_assert(values::internal::make_complex_number(3.f, 4.f) == std::complex<float>{3, 4});
}


#include "values/functions/internal/update_real_part.hpp"

TEST(values, update_real_part)
{
  static_assert(values::internal::update_real_part(std::complex{3.5, 4.5}, 5.5) == std::complex{5.5, 4.5});
  static_assert(values::internal::update_real_part(std::complex{3, 4}, 5.2) == std::complex{5, 4}); // truncation occurs
}


#include "values/functions/internal/near.hpp"

TEST(values, near_complex)
{
  static_assert(values::internal::near(std::complex<double>{3, 4}, std::complex<double>{3 + 1e-9, 4 + 1e-9}, 1e-6));
  static_assert(values::internal::near(std::complex<double>{3, 4}, std::complex<double>{3 - 1e-9, 4 - 1e-9}, 1e-6));
  static_assert(values::internal::near(std::complex<double>{3, 4}, std::complex<double>{3 - 1e-9, 4 - 1e-9}, std::complex<double>{1e-6, 1e-6}));
  static_assert(not values::internal::near(std::complex<double>{3, 4}, std::complex<double>{3 + 1e-4, 4 - 1e-9}, 1e-6));
  static_assert(not values::internal::near(std::complex<double>{3, 4}, std::complex<double>{3 + 1e-9, 4 - 1e-4}, 1e-6));
  static_assert(not values::internal::near(std::complex<double>{3, 4}, std::complex<double>{3 + 1e-9, 4 - 1e-4}, std::complex<double>{-1e-6, -1e-6}));

  static_assert(values::internal::near<2>(std::complex<double>{3, 4}, std::complex<double>{3 + std::numeric_limits<double>::epsilon(), 4 + std::numeric_limits<double>::epsilon()}));
  static_assert(values::internal::near<2>(std::complex<double>{3, 4}, std::complex<double>{3 - std::numeric_limits<double>::epsilon(), 4 - std::numeric_limits<double>::epsilon()}));
  static_assert(not values::internal::near<2>(std::complex<double>{3, 4}, std::complex<double>{3 + 3 * std::numeric_limits<double>::epsilon(), 4 + std::numeric_limits<double>::epsilon()}));
  static_assert(not values::internal::near<2>(std::complex<double>{3, 4}, std::complex<double>{3 - std::numeric_limits<double>::epsilon(), 4 - 3 * std::numeric_limits<double>::epsilon()}));
}

