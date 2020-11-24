/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_SPECIAL_MATRIX_ARITHMETIC_HPP
#define OPENKALMAN_EIGEN3_SPECIAL_MATRIX_ARITHMETIC_HPP

namespace OpenKalman::Eigen3
{
  // ---------- //
  //  addition  //
  // ---------- //

  /*
   * self-adjoint + self-adjoint
   */
#ifdef __cpp_concepts
  template<Eigen3::eigen_self_adjoint_expr Arg1, Eigen3::eigen_self_adjoint_expr Arg2> requires
    (not diagonal_matrix<Arg1>) and (not diagonal_matrix<Arg2>) and
    (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    Eigen3::eigen_self_adjoint_expr<Arg1> and Eigen3::eigen_self_adjoint_expr<Arg2> and
    (not diagonal_matrix<Arg1>) and (not diagonal_matrix<Arg2>) and
    (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension), int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr((Eigen3::lower_storage_triangle<Arg1> and Eigen3::lower_storage_triangle<Arg2>) or
      (Eigen3::upper_storage_triangle<Arg1> and Eigen3::upper_storage_triangle<Arg2>))
    {
      auto ret = MatrixTraits<Arg1>::make(
        nested_matrix(std::forward<Arg1&&>(arg1)) + nested_matrix(std::forward<Arg2&&>(arg2)));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
    else
    {
      auto ret = MatrixTraits<Arg1>::make(
        nested_matrix(std::forward<Arg1>(arg1)) + adjoint(nested_matrix(std::forward<Arg2>(arg2))));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
  }


  /*
   * triangular + triangular
   */
#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg1, Eigen3::eigen_triangular_expr Arg2> requires
  (not diagonal_matrix<Arg1>) and (not diagonal_matrix<Arg2>) and
    (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    Eigen3::eigen_triangular_expr<Arg1> and Eigen3::eigen_triangular_expr<Arg2> and
      not diagonal_matrix<Arg1> and not diagonal_matrix<Arg2> and
      (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension), int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(lower_triangular_matrix<Arg1> == lower_triangular_matrix<Arg2>)
    {
      auto ret = MatrixTraits<Arg1>::make(nested_matrix(std::forward<Arg1>(arg1)) + nested_matrix(std::forward<Arg2>(arg2)));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
    else
    {
      auto ret = make_native_matrix(std::forward<Arg1>(arg1));
      constexpr auto mode = upper_triangular_matrix<Arg2> ? Eigen::Upper : Eigen::Lower;
      ret.template triangularView<mode>() += nested_matrix(std::forward<Arg2>(arg2));
      return ret;
    }
  }


  /*
   * ((self-adjoint or triangular) + diagonal) or (diagonal + (self-adjoint or triangular))
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    (((Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>) and diagonal_matrix<Arg2> ) or
      (diagonal_matrix<Arg1> and (Eigen3::eigen_self_adjoint_expr<Arg2> or Eigen3::eigen_triangular_expr<Arg2>))) and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and (not identity_matrix<Arg1>) and
    (not identity_matrix<Arg2>) and (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>)
    {
      auto ret = MatrixTraits<Arg1>::make(nested_matrix(std::forward<Arg1>(arg1)) + std::forward<Arg2>(arg2));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
    else
    {
      auto ret = MatrixTraits<Arg2>::make(std::forward<Arg1>(arg1) + nested_matrix(std::forward<Arg2>(arg2)));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
  }


  /*
   * ((self-adjoint or triangular) + identity) or (identity + (self-adjoint or triangular))
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    (((Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>) and identity_matrix<Arg2>) or
      (identity_matrix<Arg1> and (Eigen3::eigen_self_adjoint_expr<Arg2> or Eigen3::eigen_triangular_expr<Arg2>))) and
    (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>)
    {
      auto ret = MatrixTraits<Arg1>::make(nested_matrix(std::forward<Arg1>(arg1)) + std::forward<Arg2>(arg2));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
    else
    {
      auto ret = MatrixTraits<Arg2>::make(std::forward<Arg1>(arg1) + nested_matrix(std::forward<Arg2>(arg2)));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
  }


  /*
   * ((self-adjoint or triangular) + zero) or (zero + (self-adjoint or triangular))
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    (((Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>) and zero_matrix<Arg2>) or
      (zero_matrix<Arg1> and (Eigen3::eigen_self_adjoint_expr<Arg2> or Eigen3::eigen_triangular_expr<Arg2>))) and
    (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  constexpr decltype(auto) operator+(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return std::forward<Arg2>(arg2);
    }
  }


  // ------------- //
  //  subtraction  //
  // ------------- //

  /*
   * self-adjoint - self-adjoint
   */
#ifdef __cpp_concepts
  template<Eigen3::eigen_self_adjoint_expr Arg1, Eigen3::eigen_self_adjoint_expr Arg2> requires
    (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<Eigen3::eigen_self_adjoint_expr<Arg1> and Eigen3::eigen_self_adjoint_expr<Arg2> and
      (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension), int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr((Eigen3::lower_storage_triangle<Arg1> and Eigen3::lower_storage_triangle<Arg2>) or
      (Eigen3::upper_storage_triangle<Arg1> and Eigen3::upper_storage_triangle<Arg2>))
    {
      auto ret = MatrixTraits<Arg1>::make(
        nested_matrix(std::forward<Arg1>(arg1)) - nested_matrix(std::forward<Arg2>(arg2)));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
    else
    {
      auto ret = MatrixTraits<Arg1>::make(
        nested_matrix(std::forward<Arg1>(arg1)) - adjoint(nested_matrix(std::forward<Arg2>(arg2))));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
  }


  /*
   * triangular - triangular
   */
#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr Arg1, Eigen3::eigen_triangular_expr Arg2> requires
    (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<Eigen3::eigen_triangular_expr<Arg1> and Eigen3::eigen_triangular_expr<Arg2> and
      (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension), int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(OpenKalman::internal::same_triangle_type_as<Arg1, Arg2>)
    {
      auto ret = MatrixTraits<Arg1>::make(nested_matrix(std::forward<Arg1>(arg1)) - nested_matrix(std::forward<Arg2>(arg2)));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
    else
    {
      auto ret = make_native_matrix(std::forward<Arg1>(arg1));
      constexpr auto mode = upper_triangular_matrix<Arg2> ? Eigen::Upper : Eigen::Lower;
      ret.template triangularView<mode>() -= nested_matrix(std::forward<Arg2>(arg2));
      return ret;
    }
  }


  /*
   * ((self-adjoint or triangular) - diagonal) or (diagonal -(self-adjoint or triangular))
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    (((Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>) and diagonal_matrix<Arg2>) or
      (diagonal_matrix<Arg1> and (Eigen3::eigen_self_adjoint_expr<Arg2> or Eigen3::eigen_triangular_expr<Arg2>))) and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and (not identity_matrix<Arg1>)
    and (not identity_matrix<Arg2>) and (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    if constexpr(Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>)
    {
      auto ret = MatrixTraits<Arg1>::make(nested_matrix(std::forward<Arg1>(arg1)) - std::forward<Arg2>(arg2));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
    else
    {
      auto ret = MatrixTraits<Arg2>::make(std::forward<Arg1>(arg1) - nested_matrix(std::forward<Arg2>(arg2)));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
  }


  /*
   * ((self-adjoint or triangular) - identity) or (identity -(self-adjoint or triangular))
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    (((Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>) and identity_matrix<Arg2>) or
      (identity_matrix<Arg1> and (Eigen3::eigen_self_adjoint_expr<Arg2> or Eigen3::eigen_triangular_expr<Arg2>))) and
    (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>)
    {
      auto ret = MatrixTraits<Arg1>::make(nested_matrix(std::forward<Arg1>(arg1)) - std::forward<Arg2>(arg2));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
    else
    {
      auto ret = MatrixTraits<Arg2>::make(std::forward<Arg1>(arg1) - nested_matrix(std::forward<Arg2>(arg2)));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
  }


  /*
   * ((self-adjoint or triangular) - zero) or (zero - (self-adjoint or triangular))
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    (((Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>) and zero_matrix<Arg2>) or
      (zero_matrix<Arg1> and (Eigen3::eigen_self_adjoint_expr<Arg2> or Eigen3::eigen_triangular_expr<Arg2>))) and
    (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return MatrixTraits<Arg2>::make(make_self_contained<Arg2>(-nested_matrix(std::forward<Arg2>(arg2))));
    }
  }


  // ---------- //
  //  negation  //
  // ---------- //

#ifdef __cpp_concepts
  template<typename Arg> requires Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<
    Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>, int> = 0>
#endif
  inline auto operator-(Arg&& arg)
  {
    return MatrixTraits<Arg>::make(make_self_contained<Arg>(-nested_matrix(std::forward<Arg>(arg))));
  }


  // ----------------------- //
  //  scalar multiplication  //
  // ----------------------- //

  /*
   * (self-adjoint or triangular) * scalar
   */
#ifdef __cpp_concepts
  template<typename Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S> requires
    Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>
#else
  template<typename Arg, typename S, std::enable_if_t<
    (Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>) and
    std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto operator*(Arg&& arg, const S scale) noexcept
  {
    auto ret = MatrixTraits<Arg>::make(nested_matrix(std::forward<Arg>(arg)) * scale);
    return make_self_contained<Arg>(std::move(ret));
  }


  /*
   * scalar * (self-adjoint or triangular)
   */
#ifdef __cpp_concepts
  template<typename Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S> requires
    Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>
#else
  template<typename Arg, typename S, std::enable_if_t<
    (Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>) and
    std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto operator*(const S scale, Arg&& arg) noexcept
  {
    auto ret = MatrixTraits<Arg>::make(scale * nested_matrix(std::forward<Arg>(arg)));
    return make_self_contained<Arg>(std::move(ret));
  }


  // ----------------- //
  //  scalar division  //
  // ----------------- //

  /*
   * (self-adjoint or triangular) / scalar
   */
#ifdef __cpp_concepts
  template<typename Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S> requires
    Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>
#else
  template<typename Arg, typename S, std::enable_if_t<
    (Eigen3::eigen_self_adjoint_expr<Arg> or Eigen3::eigen_triangular_expr<Arg>) and
    std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto operator/(Arg&& arg, const S scale) noexcept
  {
    auto ret = MatrixTraits<Arg>::make(nested_matrix(std::forward<Arg>(arg)) / scale);
    return make_self_contained<Arg>(std::move(ret));
  }


  // ----------------------- //
  //  matrix multiplication  //
  // ----------------------- //

  /*
   * (self-adjoint or triangular) * (self-adjoint or triangular)
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    ((Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>) and
      (Eigen3::eigen_self_adjoint_expr<Arg2> or Eigen3::eigen_triangular_expr<Arg2>)) and
    (not diagonal_matrix<Arg1>) and (not diagonal_matrix<Arg2>) and
    (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(OpenKalman::internal::same_triangle_type_as<Arg1, Arg2>)
    {
      auto m2 = make_native_matrix(std::forward<Arg2>(arg2));
      auto b = make_self_contained(std::forward<Arg1>(arg1).nested_view() * std::move(m2));
      return MatrixTraits<Arg1>::make(std::move(b));
    }
    else
    {
      auto m2 = make_native_matrix(std::forward<Arg2>(arg2));
      return make_self_contained(std::forward<Arg1>(arg1).nested_view() * std::move(m2));
    }
  }


  /*
   * (self-adjoint * diagonal) or (diagonal * self-adjoint)
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    ((Eigen3::eigen_self_adjoint_expr<Arg1> and diagonal_matrix<Arg2>) or
      (diagonal_matrix<Arg1> and Eigen3::eigen_self_adjoint_expr<Arg2>)) and
    (not identity_matrix<Arg1>) and (not identity_matrix<Arg2>) and (not zero_matrix<Arg1>) and
    (not zero_matrix<Arg2>) and (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(Eigen3::eigen_self_adjoint_expr<Arg1>)
    {
      auto ret = std::forward<Arg1>(arg1).nested_view() * std::forward<Arg2>(arg2);
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
    else
    {
      auto ret = std::forward<Arg1>(arg1) * std::forward<Arg2>(arg2).nested_view();
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
  }


  /*
   * (triangular * diagonal) or (diagonal * triangular)
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    ((Eigen3::eigen_triangular_expr<Arg1> and diagonal_matrix<Arg2>) or
      (diagonal_matrix<Arg1> and Eigen3::eigen_triangular_expr<Arg2>)) and
    (not identity_matrix<Arg1>) and (not identity_matrix<Arg2>) and (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>)
    and (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(Eigen3::eigen_triangular_expr<Arg1>)
    {
      auto ret = MatrixTraits<Arg1>::make(std::forward<Arg1>(arg1).nested_view() * std::forward<Arg2>(arg2));
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
    else
    {
      auto ret = MatrixTraits<Arg2>::make(std::forward<Arg1>(arg1) * std::forward<Arg2>(arg2).nested_view());
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
  }


  /*
   * ((self-adjoint or triangular) * identity) or (identity * (self-adjoint or triangular))
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    (((Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>) and identity_matrix<Arg2>) or
      (identity_matrix<Arg1> and (Eigen3::eigen_self_adjoint_expr<Arg2> or Eigen3::eigen_triangular_expr<Arg2>))) and
    (MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return std::forward<Arg2>(arg2);
    }
  }


  /*
   * ((self-adjoint or triangular) * zero) or (zero * (self-adjoint or triangular))
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    (((Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>) and zero_matrix<Arg2>) or
      (zero_matrix<Arg1> and (Eigen3::eigen_self_adjoint_expr<Arg2> or Eigen3::eigen_triangular_expr<Arg2>))) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::dimension)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {
    constexpr auto rows = MatrixTraits<Arg1>::dimension;
    constexpr auto cols = MatrixTraits<Arg2>::columns;
    using B = native_matrix_t<Arg1, rows, cols>;
    return Eigen3::ZeroMatrix<B>();
  }


  /*
   * ((self-adjoint or triangular) * native-eigen) or (native-eigen + (self-adjoint or triangular))
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    (((Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>) and Eigen3::eigen_matrix<Arg2>) or
      (Eigen3::eigen_matrix<Arg1> and (Eigen3::eigen_self_adjoint_expr<Arg2> or Eigen3::eigen_triangular_expr<Arg2>))) and
    (not identity_matrix<Arg1>) and (not identity_matrix<Arg2>) and (not zero_matrix<Arg1>) and
    (not zero_matrix<Arg2>) and (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::dimension)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(Eigen3::eigen_self_adjoint_expr<Arg1> or Eigen3::eigen_triangular_expr<Arg1>)
    {
      auto ret = std::forward<Arg1>(arg1).nested_view() * std::forward<Arg2>(arg2);
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
    else
    {
      auto ret = std::forward<Arg1>(arg1) * std::forward<Arg2>(arg2).nested_view();
      return make_self_contained<Arg1, Arg2>(std::move(ret));
    }
  }


} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_SPECIAL_MATRIX_ARITHMETIC_HPP
