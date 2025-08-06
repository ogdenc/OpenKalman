/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to Eigen::TensorFixedSize.
 */

#ifndef OPENKALMAN_EIGEN_TRAITS_TENSORFIXEDSIZE_HPP
#define OPENKALMAN_EIGEN_TRAITS_TENSORFIXEDSIZE_HPP


namespace OpenKalman::interface
{
  template<typename S, typename Dims, int options, typename IndexType>
  struct indexible_object_traits<Eigen::TensorFixedSize<S, Dims, options, IndexType>>
    : Eigen3::indexible_object_traits_tensor_base<Eigen::TensorFixedSize<S, Dims, options, IndexType>>
  {
  private:

    using Base = Eigen3::indexible_object_traits_tensor_base<Eigen::TensorFixedSize<S, Dims, options, IndexType>>;

  public:

    template<typename Arg, typename N>
    static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
    {
      if constexpr (values::fixed<N>)
        return std::integral_constant<std::size_t, Eigen::internal::get<n, typename Dims::Base>::value>{};
      else
        return static_cast<std::size_t>(arg.dimension(n));
    }

    // nested_object() not defined

    // get_constant() not defined

    // get_constant_diagonal() not defined


#ifdef __cpp_lib_concepts
    template<typename Arg, std::convertible_to<IndexType>...I> requires (sizeof...(I) == Dims::count)
#else
    template<typename Arg, typename...I, std::enable_if_t<(stdcompat::convertible_to<I, IndexType> and ...) and
      (sizeof...(I) == Dims::count), int> = 0>
#endif
    static constexpr decltype(auto)
    get(Arg&& arg, I...i)
    {
      if constexpr ((Eigen::internal::traits<std::decay_t<Arg>>::Flags & Eigen::LvalueBit) != 0)
        return std::forward<Arg>(arg).coeffRef(static_cast<IndexType>(i)...);
      else
        return std::forward<Arg>(arg).coeff(static_cast<IndexType>(i)...);
    }


#ifdef __cpp_lib_concepts
    template<typename Arg, std::convertible_to<IndexType>...I> requires (sizeof...(I) == Dims::count) and
      ((Eigen::internal::traits<std::decay_t<Arg>>::Flags & Eigen::LvalueBit) != 0x0)
#else
    template<typename Arg, typename...I, std::enable_if_t<(std::is_convertible_v<I, IndexType> and ...) and
      (sizeof...(I) == Dims::count) and ((Eigen::internal::traits<std::decay_t<Arg>>::Flags & Eigen::LvalueBit) != 0x0), int> = 0>
#endif
    static void
    set(Arg& arg, const scalar_type_of_t<Arg>& s, I...i)
    {
      arg.coeffRef(static_cast<IndexType>(i)...) = s;
    }


    static constexpr bool is_writable = true;

    template<typename Arg>
    static constexpr auto * const
    raw_data(Arg& arg) { return arg.data(); }

    static constexpr Layout layout = options & Eigen::RowMajor ? Layout::right : Layout::left;

  };

} // namespace OpenKalman::interface

#endif //OPENKALMAN_EIGEN_TRAITS_TENSORFIXEDSIZE_HPP
