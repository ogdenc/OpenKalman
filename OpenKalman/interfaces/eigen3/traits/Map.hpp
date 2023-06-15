/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to Eigen::Map.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_MAP_HPP
#define OPENKALMAN_EIGEN3_TRAITS_MAP_HPP

#include <type_traits>


namespace OpenKalman::interface
{
#ifndef __cpp_concepts
  template<typename PlainObjectType, int MapOptions, typename StrideType>
  struct IndexTraits<Eigen::Map<PlainObjectType, MapOptions, StrideType>>
    : detail::IndexTraits_Eigen_default<Eigen::Map<PlainObjectType, MapOptions, StrideType>> {};
#endif


  template<typename PlainObjectType, int MapOptions, typename StrideType>
  struct Dependencies<Eigen::Map<PlainObjectType, MapOptions, StrideType>>
  {
  private:
    using M = Eigen::Map<PlainObjectType, MapOptions, StrideType>;
  public:
    static constexpr bool has_runtime_parameters =
      M::RowsAtCompileTime == Eigen::Dynamic or M::ColsAtCompileTime == Eigen::Dynamic or
      M::OuterStrideAtCompileTime == Eigen::Dynamic or M::InnerStrideAtCompileTime == Eigen::Dynamic;

    // Map is not self-contained in any circumstances.
    using type = std::tuple<decltype(*std::declval<typename M::PointerType>())>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return *std::forward<Arg>(arg).data();
    }
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_MAP_HPP
