/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "patterns/patterns.hpp"
#include "linear-algebra/functions/copy_from.hpp"
#include "linear-algebra/tests/tests.hpp"
#include "linear-algebra/interfaces/stl/mdspan-object.hpp"
#include "linear-algebra/interfaces/stl/array-object.hpp"
#include "linear-algebra/functions/to_diagonal.hpp"
#include "linear-algebra/functions/to_triangular.hpp"
#include "linear-algebra/functions/to_hermitian.hpp"

using namespace OpenKalman;

using namespace OpenKalman;
using namespace OpenKalman::test;
using namespace OpenKalman::patterns;

using stdex::numbers::pi;

namespace
{
  using A23 = double[2][3];
  using A32 = double[3][2];
  using A23c = std::complex<double>[2][3];
  using A32c = std::complex<double>[3][2];
  using M11 = stdex::mdspan<double, stdex::extents<std::size_t, 1, 1>>;
  using M31 = stdex::mdspan<double, stdex::extents<std::size_t, 3, 1>>;
  using M23 = stdex::mdspan<double, stdex::extents<std::size_t, 2, 3>>;
  using M32 = stdex::mdspan<double, stdex::extents<std::size_t, 3, 2>>;
  using M23c = stdex::mdspan<std::complex<double>, stdex::extents<std::size_t, 2, 3>>;
  using M32c = stdex::mdspan<std::complex<double>, stdex::extents<std::size_t, 3, 2>>;
  using M123 = stdex::mdspan<double, stdex::extents<std::size_t, 1, 2, 3>>;
  using M213 = stdex::mdspan<double, stdex::extents<std::size_t, 2, 1, 3>>;
  using M123c = stdex::mdspan<std::complex<double>, stdex::extents<std::size_t, 1, 2, 3>>;
  using M213c = stdex::mdspan<std::complex<double>, stdex::extents<std::size_t, 2, 1, 3>>;

  using MD33 = stdex::mdspan<
    std::complex<double>,
    stdex::extents<std::size_t, 3, 3>,
    interface::layout_to_diagonal<typename M31::layout_type, stdex::extents<std::size_t, 3, 1>>,
    interface::to_diagonal_accessor<stdex::default_accessor<std::complex<double>>>>;

  using MTL23 = stdex::mdspan<
    std::complex<double>,
    stdex::extents<std::size_t, 2, 3>,
    interface::layout_triangle_partition<typename M23::layout_type, triangle_type::lower>,
    interface::triangular_accessor<stdex::default_accessor<std::complex<double>>>>;

  using MH22 = stdex::mdspan<
    std::complex<double>,
    stdex::extents<std::size_t, 2, 2>,
    interface::layout_triangle_partition<typename M31::layout_type, triangle_type::upper>,
    interface::hermitian_accessor<stdex::default_accessor<std::complex<double>>>>;

}


#include "linear-algebra/functions/attach_patterns.hpp"
#include "linear-algebra/functions/make_constant_diagonal.hpp"
#include "linear-algebra/functions/conjugate.hpp"
#include "linear-algebra/concepts/triangular_matrix.hpp"
#include "linear-algebra/concepts/diagonal_matrix.hpp"

TEST(stl_interfaces, conjugate)
{
  auto p_pd = std::tuple{Polar{}, std::tuple{Polar{}, Distance{}}};
  using P_PD = decltype(p_pd);

  A23 a23 {{1, 2, 3}, {4, 5, 6}};
  auto m23 = get_mdspan(a23);
  EXPECT_TRUE(is_near(conjugate(a23), a23));
  EXPECT_TRUE(is_near(conjugate(m23), m23));

  static_assert(diagonal_matrix<MD33>);
  static_assert(triangular_matrix<MTL23, triangle_type::lower>);
  static_assert(hermitian_matrix<MH22>);

  static_assert(diagonal_matrix<decltype(conjugate(std::declval<MD33>()))>);
  static_assert(triangular_matrix<decltype(conjugate(std::declval<MTL23>())), triangle_type::lower>);
  static_assert(hermitian_matrix<decltype(conjugate(std::declval<MH22>()))>);

  auto pa23 = attach_patterns(a23, p_pd);
  auto pm23 = attach_patterns(m23, p_pd);
  static_assert(stdex::same_as<pattern_collection_type_of_t<decltype(conjugate(pm23))>, P_PD>);
  static_assert(stdex::same_as<pattern_collection_type_of_t<decltype(conjugate(pa23))>, P_PD>);
  EXPECT_TRUE(is_near(conjugate(pm23), pm23));
  EXPECT_TRUE(is_near(conjugate(pa23), pa23));

  static_assert(stdex::same_as<pattern_collection_type_of_t<decltype(conjugate(make_constant(std::complex{1.,.1}, p_pd)))>, P_PD>);
  static_assert(stdex::same_as<pattern_collection_type_of_t<decltype(conjugate(std::declval<pattern_adapter<M23, P_PD>&>()))>, P_PD>);
  static_assert(stdex::same_as<pattern_collection_type_of_t<decltype(conjugate(std::declval<pattern_adapter<A23&, P_PD>&>()))>, P_PD>);
  static_assert(stdex::same_as<pattern_collection_type_of_t<decltype(conjugate(std::declval<pattern_adapter<M23c, P_PD>&>()))>, P_PD>);
  static_assert(stdex::same_as<pattern_collection_type_of_t<decltype(conjugate(std::declval<pattern_adapter<A23c&, P_PD>&>()))>, P_PD>);

  EXPECT_TRUE(is_near(conjugate(make_constant(std::complex{1.,.1}, p_pd)), make_constant(std::complex{1.,-.1}, p_pd)));

  EXPECT_TRUE(is_near(conjugate(make_constant_diagonal(std::complex{1.,.1}, std::tuple{Polar{}})), make_constant_diagonal(std::complex{1.,-.1}, std::tuple{Polar{}})));

  A23c a23c {{std::complex{1.,.1}, std::complex{2.,.2}, std::complex{3.,.3}},
            {std::complex{4.,-.4}, std::complex{5.,-.5}, std::complex{6.,-.6}}};
  M23c m23c = get_mdspan(a23c);
  A23c a23conj {{std::complex{1.,-.1}, std::complex{2.,-.2}, std::complex{3.,-.3}},
            {std::complex{4.,.4}, std::complex{5.,.5}, std::complex{6.,.6}}};
  M23c m23conj = get_mdspan(a23conj);

  std::complex<double> a2c[2] {std::complex{1.,-.1}, std::complex{5.,.5}};
  EXPECT_TRUE(is_near(conjugate(to_diagonal(a2c, p_pd)), to_diagonal(conjugate(a2c), p_pd)));
  EXPECT_TRUE(is_near(diagonal_of(conjugate(to_diagonal(a2c, p_pd))), diagonal_of(a23c)));
  EXPECT_TRUE(is_near(conjugate(diagonal_of(a23c)), a2c));
  EXPECT_TRUE(is_near(diagonal_of(conjugate(a23c)), a2c));

  EXPECT_TRUE(is_near(conjugate(m23c), m23conj));
  EXPECT_TRUE(is_near(conjugate(m23conj), m23c));
  EXPECT_TRUE(is_near(conjugate(a23c), a23conj));
  EXPECT_TRUE(is_near(conjugate(a23conj), a23c));

  auto pa23c = attach_patterns(a23c, p_pd);
  auto pm23c = attach_patterns(m23c, p_pd);
  auto pa23conj = attach_patterns(a23conj, p_pd);
  auto pm23conj = attach_patterns(m23conj, p_pd);
  static_assert(stdex::same_as<pattern_collection_type_of_t<decltype(conjugate(pm23c))>, P_PD>);
  static_assert(stdex::same_as<pattern_collection_type_of_t<decltype(conjugate(pa23c))>, P_PD>);
  static_assert(stdex::same_as<pattern_collection_type_of_t<decltype(conjugate(pm23conj))>, P_PD>);
  static_assert(stdex::same_as<pattern_collection_type_of_t<decltype(conjugate(pa23conj))>, P_PD>);
  EXPECT_TRUE(is_near(conjugate(pm23c), pm23conj));
  EXPECT_TRUE(is_near(conjugate(pm23conj), pm23c));
  EXPECT_TRUE(is_near(conjugate(pa23c), pa23conj));
  EXPECT_TRUE(is_near(conjugate(pa23conj), pa23c));

  auto pa2c = attach_patterns(a2c, std::tuple{Polar{}});
  static_assert(stdex::same_as<pattern_collection_type_of_t<decltype(conjugate(pa2c))>, std::tuple<Polar<>>>);
  EXPECT_TRUE(is_near(conjugate(diagonal_of(pa23c)), pa2c));
  EXPECT_TRUE(is_near(diagonal_of(conjugate(pa23c)), pa2c));
}

#include "linear-algebra/functions/transpose.hpp"

TEST(stl_interfaces, transpose)
{
  auto p_pd = std::tuple{Polar{}, std::tuple{Polar{}, Distance{}}};
  using P_PD = decltype(p_pd);

  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(transpose(make_constant(std::complex{1.,.1}, p_pd)))>, std::tuple<std::tuple<Polar<>, Distance>, Polar<>>>);
  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(transpose(std::declval<pattern_adapter<M23, P_PD>&>()))>, std::tuple<std::tuple<Polar<>, Distance>, Polar<>>>);
  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(transpose(std::declval<pattern_adapter<A23&, P_PD>&>()))>, std::tuple<std::tuple<Polar<>, Distance>, Polar<>>>);
  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(transpose(std::declval<pattern_adapter<M23c, P_PD>&>()))>, std::tuple<std::tuple<Polar<>, Distance>, Polar<>>>);
  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(transpose(std::declval<pattern_adapter<A23c&, P_PD>&>()))>, std::tuple<std::tuple<Polar<>, Distance>, Polar<>>>);

  EXPECT_TRUE(is_near(transpose(make_constant(1.1, p_pd)), make_constant(1.1, std::tuple{std::tuple{Polar{}, Distance{}}, Polar{}})));

  EXPECT_TRUE(is_near(transpose(make_constant_diagonal(1.1, p_pd)), make_constant_diagonal(1.1, std::tuple{std::tuple{Polar{}, Distance{}}, Polar{}})));

  static_assert(diagonal_matrix<decltype(transpose(std::declval<MD33>()))>);
  static_assert(triangular_matrix<decltype(transpose(std::declval<MTL23>())), triangle_type::upper>);
  static_assert(hermitian_matrix<decltype(transpose(std::declval<MH22>()))>);

  std::complex<double> a2c[2] {std::complex{1.,-.1}, std::complex{5.,-.5}};
  EXPECT_TRUE(is_near(transpose(to_diagonal(a2c)), to_diagonal(a2c)));
  EXPECT_TRUE(is_near(transpose(to_diagonal(a2c, p_pd)), to_diagonal(a2c, patterns::views::transpose<>(p_pd))));

  A23 a23 {{1, 2, 3}, {4, 5, 6}};
  A32 a32 {{1, 4}, {2, 5}, {3, 6}};
  M23 m23 = get_mdspan(a23);
  auto a32lin = std::array{1., 4., 2., 5., 3., 6.};
  M32 m32 {a32lin.data()};
  EXPECT_TRUE(is_near(transpose(a23), a32));
  EXPECT_TRUE(is_near(transpose(m23), m32));
  EXPECT_TRUE(is_near(transpose(a32), m23));
  EXPECT_TRUE(is_near(transpose(m32), a23));

  auto pa23 = attach_patterns(a23, p_pd);
  auto pm23 = attach_patterns(m23, p_pd);
  auto pa32 = attach_patterns(a32, std::tuple{std::tuple{Polar{}, Distance{}}, Polar{}});
  auto pm32 = attach_patterns(m32, std::tuple{std::tuple{Polar{}, Distance{}}, Polar{}});
  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(transpose(pm23))>, std::tuple<std::tuple<Polar<>, Distance>, Polar<>>>);
  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(transpose(pa23))>, std::tuple<std::tuple<Polar<>, Distance>, Polar<>>>);
  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(transpose(pm32))>, P_PD>);
  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(transpose(pa32))>, P_PD>);
  EXPECT_TRUE(is_near(transpose(pm23), pm32));
  EXPECT_TRUE(is_near(transpose(pa23), pa32));
  EXPECT_TRUE(is_near(transpose(pm23), pa32));
  EXPECT_TRUE(is_near(transpose(pa23), pm32));

  M123 m123 {a23[0]};
  EXPECT_TRUE(is_near(transpose<0, 2>(m32), m123));
  EXPECT_TRUE(is_near(transpose<0, 2>(m123), m32));
  EXPECT_TRUE(is_near(transpose<0, 2>(pm32), attach_patterns(m123, std::tuple{Dimensions<1>{}, Polar{}, std::tuple{Polar{}, Distance{}}})));
  EXPECT_TRUE(is_near(transpose<0, 2>(attach_patterns(m123, std::tuple{Dimensions<1>{}, Polar{}, std::tuple{Polar{}, Distance{}}})), pm32));

  M213 m213 {a23[0]};
  EXPECT_TRUE(is_near(transpose<1, 2>(m23), m213));
  EXPECT_TRUE(is_near(transpose<1, 2>(m213), m23));
  EXPECT_TRUE(is_near(transpose<1, 2>(pm23), attach_patterns(m213, std::tuple{Polar{}, Dimensions<1>{}, std::tuple{Polar{}, Distance{}}})));
  EXPECT_TRUE(is_near(transpose<1, 2>(attach_patterns(m213, std::tuple{Polar{}, Dimensions<1>{}, std::tuple{Polar{}, Distance{}}})), pm23));

  EXPECT_TRUE(is_near(transpose<2, 3>(m23), m23));
  EXPECT_TRUE(is_near(transpose<2, 3>(pm23), pm23));
}

#include "linear-algebra/functions/conjugate_transpose.hpp"

TEST(stl_interfaces, conjugate_transpose)
{
  auto p_pd = std::tuple{Polar{}, std::tuple{Polar{}, Distance{}}};
  using P_PD = decltype(p_pd);

  double a2[2] {1., 5.};
  EXPECT_TRUE(is_near(conjugate_transpose(to_diagonal(a2)), to_diagonal(a2)));

  A23 a23 {{1, 2, 3}, {4, 5, 6}};
  A32 a32 {{1, 4}, {2, 5}, {3, 6}};
  M23 m23 = get_mdspan(a23);
  auto a32lin = std::array{1., 4., 2., 5., 3., 6.};
  M32 m32 {a32lin.data()};
  EXPECT_TRUE(is_near(conjugate_transpose(a23), a32));
  EXPECT_TRUE(is_near(conjugate_transpose(m23), m32));
  EXPECT_TRUE(is_near(conjugate_transpose(a32), m23));
  EXPECT_TRUE(is_near(conjugate_transpose(m32), a23));

  EXPECT_TRUE(is_near(conjugate_transpose(make_constant(std::complex{1.,0.}, p_pd)), make_constant(std::complex{1.,0.}, std::tuple{std::tuple{Polar{}, Distance{}}, Polar{}})));
  EXPECT_TRUE(is_near(conjugate_transpose(make_constant_diagonal(std::complex{1.,0.}, p_pd)), make_constant_diagonal(std::complex{1.,0.}, std::tuple{std::tuple{Polar{}, Distance{}}, Polar{}})));

  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(conjugate_transpose(make_constant(std::complex{1.,.1}, p_pd)))>, std::tuple<std::tuple<Polar<>, Distance>, Polar<>>>);
  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(conjugate_transpose(std::declval<pattern_adapter<M23, P_PD>&>()))>, std::tuple<std::tuple<Polar<>, Distance>, Polar<>>>);
  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(conjugate_transpose(std::declval<pattern_adapter<A23&, P_PD>&>()))>, std::tuple<std::tuple<Polar<>, Distance>, Polar<>>>);
  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(conjugate_transpose(std::declval<pattern_adapter<M23c, P_PD>&>()))>, std::tuple<std::tuple<Polar<>, Distance>, Polar<>>>);
  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(conjugate_transpose(std::declval<pattern_adapter<A23c&, P_PD>&>()))>, std::tuple<std::tuple<Polar<>, Distance>, Polar<>>>);

  static_assert(diagonal_matrix<decltype(conjugate_transpose(std::declval<MD33>()))>);
  static_assert(triangular_matrix<decltype(conjugate_transpose(std::declval<MTL23>())), triangle_type::upper>);
  static_assert(hermitian_matrix<decltype(conjugate_transpose(std::declval<MH22>()))>);

  EXPECT_TRUE(is_near(conjugate_transpose(make_constant(std::complex{1.,.1}, p_pd)), make_constant(std::complex{1.,-.1}, std::tuple{std::tuple{Polar{}, Distance{}}, Polar{}})));
  EXPECT_TRUE(is_near(conjugate_transpose(make_constant_diagonal(std::complex{1.,.1}, p_pd)), make_constant_diagonal(std::complex{1.,-.1}, std::tuple{std::tuple{Polar{}, Distance{}}, Polar{}})));

  std::complex<double> a2c[2] {std::complex{1.,-.1}, std::complex{5.,-.5}};
  EXPECT_TRUE(is_near(conjugate_transpose(to_diagonal(a2c)), to_diagonal(conjugate(a2c))));
  EXPECT_TRUE(is_near(conjugate_transpose(to_diagonal(a2c, p_pd)), to_diagonal(conjugate(a2c), patterns::views::transpose<>(p_pd))));

  A23c a23c {{std::complex{1.,.1}, std::complex{2.,.2}, std::complex{3.,.3}}, {std::complex{4.,.4}, std::complex{5.,.5}, std::complex{6.,.6}}};
  A32c a32c {{std::complex{1.,-.1}, std::complex{4.,-.4}}, {std::complex{2.,-.2}, std::complex{5.,-.5}}, {std::complex{3.,-.3}, std::complex{6.,-.6}}};
  M23c m23c = get_mdspan(a23c);
  M32c m32c = get_mdspan(a32c);
  EXPECT_TRUE(is_near(conjugate_transpose(a23c), a32c));
  EXPECT_TRUE(is_near(conjugate_transpose(m23c), m32c));
  EXPECT_TRUE(is_near(conjugate_transpose(a32c), m23c));
  EXPECT_TRUE(is_near(conjugate_transpose(m32c), a23c));

  auto pa23c = attach_patterns(a23c, p_pd);
  auto pm23c = attach_patterns(m23c, p_pd);
  auto pa32c = attach_patterns(a32c, std::tuple{std::tuple{Polar{}, Distance{}}, Polar{}});
  auto pm32c = attach_patterns(m32c, std::tuple{std::tuple{Polar{}, Distance{}}, Polar{}});
  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(conjugate_transpose(pm23c))>, std::tuple<std::tuple<Polar<>, Distance>, Polar<>>>);
  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(conjugate_transpose(pa23c))>, std::tuple<std::tuple<Polar<>, Distance>, Polar<>>>);
  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(conjugate_transpose(pm32c))>, P_PD>);
  static_assert(patterns::collection_compares_with<pattern_collection_type_of_t<decltype(conjugate_transpose(pa32c))>, P_PD>);
  EXPECT_TRUE(is_near(conjugate_transpose(pm23c), pm32c));
  EXPECT_TRUE(is_near(conjugate_transpose(pa23c), pa32c));
  EXPECT_TRUE(is_near(conjugate_transpose(pm23c), pa32c));
  EXPECT_TRUE(is_near(conjugate_transpose(pa23c), pm32c));

  M123c m123c {a23c[0]};
  EXPECT_TRUE(is_near(conjugate_transpose<0, 2>(m32c), m123c));
  EXPECT_TRUE(is_near(conjugate_transpose<0, 2>(m123c), m32c));
  EXPECT_TRUE(is_near(conjugate_transpose<0, 2>(pm32c), attach_patterns(m123c, std::tuple{Dimensions<1>{}, Polar{}, std::tuple{Polar{}, Distance{}}})));
  EXPECT_TRUE(is_near(conjugate_transpose<0, 2>(attach_patterns(m123c, std::tuple{Dimensions<1>{}, Polar{}, std::tuple{Polar{}, Distance{}}})), pm32c));

  A23c a23conj {{std::complex{1.,-.1}, std::complex{2.,-.2}, std::complex{3.,-.3}}, {std::complex{4.,-.4}, std::complex{5.,-.5}, std::complex{6.,-.6}}};
  M213c m213c {a23conj[0]};
  EXPECT_TRUE(is_near(conjugate_transpose<1, 2>(m23c), m213c));
  EXPECT_TRUE(is_near(conjugate_transpose<1, 2>(m213c), m23c));
  EXPECT_TRUE(is_near(conjugate_transpose<1, 2>(pm23c), attach_patterns(m213c, std::tuple{Polar{}, Dimensions<1>{}, std::tuple{Polar{}, Distance{}}})));
  EXPECT_TRUE(is_near(conjugate_transpose<1, 2>(attach_patterns(m213c, std::tuple{Polar{}, Dimensions<1>{}, std::tuple{Polar{}, Distance{}}})), pm23c));

  EXPECT_TRUE(is_near(conjugate_transpose<2, 3>(m23c), a23conj));
  EXPECT_TRUE(is_near(conjugate_transpose<2, 3>(attach_patterns(a23conj, p_pd)), pm23c));
}

