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
 * \brief Type traits as applied to Eigen::Inverse.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_INVERSE_HPP
#define OPENKALMAN_EIGEN3_TRAITS_INVERSE_HPP

#include <type_traits>


namespace OpenKalman::interface
{
#ifndef __cpp_concepts
  template<typename XprType>
  struct IndexTraits<Eigen::Inverse<XprType>> : detail::IndexTraits_Eigen_default<Eigen::Inverse<XprType>> {};
#endif


  template<typename XprType>
  struct Dependencies<Eigen::Inverse<XprType>>
  {
  private:

    using T = Eigen::Inverse<XprType>;

  public:

    static constexpr bool has_runtime_parameters = false;
    using type = std::tuple<typename T::XprTypeNested>;

    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg)
    {
      static_assert(i == 0);
      return std::forward<Arg>(arg).nestedExpression();
    }

    template<typename Arg>
    static auto convert_to_self_contained(Arg&& arg)
    {
      using N = Eigen::Inverse<equivalent_self_contained_t<XprType>>;
      if constexpr (not std::is_lvalue_reference_v<typename N::XprTypeNested>)
        return N {make_self_contained(arg.nestedExpression())};
      else
        return make_dense_writable_matrix_from(std::forward<Arg>(arg));
    }
  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN3_TRAITS_INVERSE_HPP