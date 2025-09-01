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

#include "values/traits/value_type_of.hpp"
#include "values/traits/real_type_of.hpp"

TEST(values, interface)
{
  static_assert(stdcompat::same_as<values::value_type_of_t<std::complex<double>>, std::complex<double>>);
  static_assert(stdcompat::same_as<values::real_type_of_t<std::complex<double>>, double>);
  static_assert(stdcompat::same_as<values::real_type_of_t<double>, double>);
  static_assert(stdcompat::same_as<values::real_type_of_t<int>, double>);
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
#include "values/classes/fixed_value.hpp"

TEST(values, Fixed_complex)
{
  static_assert(not values::complex<double>);
  static_assert(values::not_complex<double>);
  static_assert(values::not_complex<int>);
  static_assert(not values::not_complex<std::complex<double>>);

  static_assert(values::complex<values::fixed_value<std::complex<double>, 3>>);
  static_assert(values::not_complex<values::fixed_value<std::complex<double>, 3>>);
  static_assert(values::not_complex<values::fixed_value<std::complex<double>, 3, 0>>);
  static_assert(not values::not_complex<values::fixed_value<std::complex<double>, 3, 1>>);

  static_assert(std::real(values::to_value_type(values::fixed_value<std::complex<double>, 3, 4>{})) == 3);
  static_assert(std::imag(values::to_value_type(values::fixed_value<std::complex<double>, 3, 4>{})) == 4);
  static_assert(values::fixed_value<std::complex<double>, 3, 4>{}() == std::complex<double>{3, 4});

  EXPECT_TRUE(test::is_near(values::fixed_value<std::complex<double>, 3, 4>{}, values::fixed_value<std::complex<double>, 4, 3>{}, 2));
  EXPECT_TRUE(test::is_near(values::fixed_value<std::complex<double>, 3, 4>{}, std::complex<double>{2, 5}, std::complex<double>{2, 2}));
  EXPECT_TRUE(test::is_near(std::complex<double>{3 - 1e-9, 4 + 1e-9}, values::fixed_value<std::complex<double>, 3, 4>{}, 1e-6));
  EXPECT_TRUE(test::is_near(values::fixed_value<std::complex<double>, 3, 0>{}, 3. - 1e-9, 1e-6));
}


#include "values/functions/internal/make_complex_number.hpp"

TEST(values, make_complex_number)
{
  static_assert(values::internal::make_complex_number<std::complex<double>>(3., 4.) == std::complex<double>{3, 4});
  static_assert(values::internal::make_complex_number<std::complex<double>>(3., 4.f) == std::complex<double>{3, 4});
  static_assert(values::internal::make_complex_number<std::complex<double>>(std::complex<float>{3, 4}) == std::complex<double>{3, 4});
  static_assert(values::internal::make_complex_number<double>(std::complex<double>{3, 4}) == std::complex<double>{3, 4});
  static_assert(values::internal::make_complex_number<double>(std::complex<float>{3, 4}) == std::complex<double>{3, 4});
  static_assert(values::internal::make_complex_number<>(3., 4.) == std::complex<double>{3, 4});
  static_assert(values::internal::make_complex_number<>(3., 4.f) == std::complex<double>{3, 4});
  static_assert(values::internal::make_complex_number<>(3.f, 4.f) == std::complex<float>{3, 4});
  static_assert(values::internal::make_complex_number<double>(3.) == std::complex<double>{3., 0.});
}


#include "values/functions/internal/update_real_part.hpp"

TEST(values, update_real_part)
{
  static_assert(values::internal::update_real_part(std::complex{3.5, 4.5}, 5.5) == std::complex{5.5, 4.5});
  static_assert(values::internal::update_real_part(std::complex{3, 4}, 5.2) == std::complex{5.2, 4.}); // truncation occurs
  static_assert(values::internal::update_real_part(values::fixed_value<std::complex<double>, 3, 4>{}, 5.5) == std::complex{5.5, 4.});
  static_assert(values::real(values::internal::update_real_part(values::fixed_value<std::complex<double>, 3, 4>{}, values::fixed_value<double, 5>{})) == 5.);
  static_assert(values::imag(values::internal::update_real_part(values::fixed_value<std::complex<double>, 3, 4>{}, values::fixed_value<double, 5>{})) == 4.);
}

#include "values/traits/complex_type_of.hpp"

TEST(values, complex_type_of)
{
  static_assert(stdcompat::same_as<values::complex_type_of_t<double>, std::complex<double>>);
  static_assert(stdcompat::same_as<values::complex_type_of_t<float>, std::complex<float>>);
  static_assert(stdcompat::same_as<values::complex_type_of_t<std::complex<double>>, std::complex<double>>);
#if __cpp_nontype_template_args >= 201911L
  static_assert(stdcompat::same_as<values::complex_type_of_t<values::fixed_value<double, 3.>>, values::fixed_value<std::complex<double>, 3., 0.>>);
#else
  static_assert(stdcompat::same_as<values::complex_type_of_t<values::fixed_value<double, 3>>, values::fixed_value<std::complex<double>, static_cast<std::intmax_t>(3), static_cast<std::intmax_t>(0)>>);
#endif
  static_assert(values::real(values::fixed_value_of_v<values::complex_type_of_t<values::fixed_value<double, 3>>>) == 3.);
  static_assert(values::imag(values::fixed_value_of_v<values::complex_type_of_t<values::fixed_value<double, 3>>>) == 0.);
  static_assert(values::fixed_value_of_v<values::complex_type_of_t<values::fixed_value<std::complex<double>, 3, 4>>> == std::complex<double>{3., 4.});
}

#include "values/functions/cast_to.hpp"

TEST(values, cast_to_complex)
{
  static_assert(values::cast_to<double>(std::complex<float>{3, 4}) == std::complex<double>{3, 4});
  static_assert(values::cast_to<std::complex<double>>(std::complex<float>{3, 4}) == std::complex<double>{3, 4});
  static_assert(values::real(values::cast_to<double>(values::fixed_value<std::complex<float>, 3, 4>{})) == 3);
  static_assert(values::imag(values::cast_to<double>(values::fixed_value<std::complex<float>, 3, 4>{})) == 4);
  static_assert(stdcompat::same_as<decltype(values::cast_to<double>(values::fixed_value<double, 4>{})), values::fixed_value<double, 4>&&>);
  static_assert(stdcompat::same_as<typename values::fixed_value_of<decltype(values::cast_to<double>(values::fixed_value<double, 4>{}))>::value_type, double>);
  static_assert(stdcompat::same_as<values::fixed_value_of<decltype(values::cast_to<int>(values::fixed_value<int, 4>{}))>::value_type, int>);
  static_assert(stdcompat::same_as<values::fixed_value_of<decltype(values::cast_to<double>(values::fixed_value<int, 4>{}))>::value_type, double>);
  static_assert(values::cast_to<double>(values::fixed_value<float, 4>{}) == 4);
  static_assert(stdcompat::same_as<values::value_type_of_t<decltype(values::cast_to<double>(values::fixed_value<float, 4>{}))>, double>);
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

  static_assert(values::internal::near(std::integral_constant<int, 4>{}, values::fixed_value<std::complex<double>, 4, 1>{}, 2));
  static_assert(values::internal::near(values::fixed_value<std::complex<double>, 3, 4>{}, values::fixed_value<std::complex<double>, 4, 5>{}, 2));
  static_assert(values::internal::near(values::fixed_value<std::complex<double>, 3, 4>{}, values::fixed_value<std::complex<double>, 4, 5>{}, values::fixed_value<std::complex<double>, 2, 2>{}));
}
