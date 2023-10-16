/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definitions for Eigen3::EigenAdapterBase
 */

#ifndef OPENKALMAN_EIGENADAPTERBASE_HPP
#define OPENKALMAN_EIGENADAPTERBASE_HPP

namespace OpenKalman::Eigen3
{
  template<typename Derived, typename NestedMatrix>
  struct EigenAdapterBase : EigenDenseBase,
      std::conditional_t<std::is_base_of_v<Eigen::ArrayBase<std::decay_t<NestedMatrix>>, std::decay_t<NestedMatrix>>,
      Eigen::ArrayBase<Derived>, Eigen::MatrixBase<Derived>>,
      Eigen::internal::no_assignment_operator // Override all Eigen assignment operators
  {

  private:

    using Base = std::conditional_t<std::is_base_of_v<Eigen::ArrayBase<std::decay_t<NestedMatrix>>, std::decay_t<NestedMatrix>>,
      Eigen::ArrayBase<Derived>, Eigen::MatrixBase<Derived>>;

  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Derived)
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(bool {has_dynamic_dimensions<NestedMatrix>})


    EigenAdapterBase() = default;

    EigenAdapterBase(const EigenAdapterBase&) = default;

    EigenAdapterBase(EigenAdapterBase&&) = default;

    ~EigenAdapterBase() = default;

    constexpr EigenAdapterBase& operator=(const EigenAdapterBase& other) { return *this; }

    constexpr EigenAdapterBase& operator=(EigenAdapterBase&& other) { return *this; }

    using typename Base::Index;


    /**
     * \internal
     * \return The number of rows at runtime.
     * \note Eigen3 requires this, particularly in Eigen::EigenBase.
     */
    constexpr Index rows() const
    {
      return get_index_dimension_of<0>(static_cast<const Derived&>(*this));
    }


    /**
     * \internal
     * \return The number of columns at runtime.
     * \note Eigen3 requires this, particularly in Eigen::EigenBase.
     */
    constexpr Index cols() const
    {
      return get_index_dimension_of<1>(static_cast<const Derived&>(*this));
    }


#ifdef __cpp_concepts
    constexpr decltype(auto) data() requires directly_accessible<NestedMatrix>
#else
    template<typename T = NestedMatrix, std::enable_if_t<directly_accessible<T>, int> = 0>
      constexpr decltype(auto)
      data()
#endif
    {
      return internal::raw_data(static_cast<const Derived&>(*this));
    }


#ifdef __cpp_concepts
    constexpr decltype(auto) data() const requires directly_accessible<NestedMatrix>
#else
    template<typename T = NestedMatrix, std::enable_if_t<directly_accessible<T>, int> = 0>
      constexpr decltype(auto)
      data() const
#endif
    {
      return internal::raw_data(static_cast<const Derived&>(*this));
    }

  private:

#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct has_outerStride : std::false_type {};

    template<typename T>
    struct has_outerStride<T, std::void_t<decltype(std::declval<T>().outerStride())>> : std::true_type {};

    template<typename T, typename = void>
    struct has_innerStride : std::false_type {};

    template<typename T>
    struct has_innerStride<T, std::void_t<decltype(std::declval<T>().innerStride())>> : std::true_type {};
#endif

  public:

    constexpr Index
    outerStride() const
    {
      if constexpr (layout_of_v<Derived> == Layout::stride)
        return std::get<Base::IsRowMajor ? 0 : 1>(strides(static_cast<const Derived&>(*this)));
      else if constexpr (Base::IsRowMajor)
        return cols();
      else
        return rows();
    }


    constexpr Index
    innerStride() const
    {
      if constexpr (layout_of_v<Derived> == Layout::stride)
        return std::get<Base::IsRowMajor ? 1 : 0>(strides(static_cast<const Derived&>(*this)));
      else
        return 1;
    }


    /**
     * \note Overrides Eigen::DenseBase<Derived>::Zero.
     * \return A matrix, of the same size and shape, containing only zero coefficients.
     */
    [[deprecated("Use make_zero_matrix_like() instead.")]]
    static constexpr auto Zero()
    {
      static_assert(not has_dynamic_dimensions<Derived>);
      return make_zero_matrix_like<Derived>();
    }


    /**
     * \note Overrides Eigen::DenseBase<Derived>::Zero.
     * \return A matrix, of the same size and shape, containing only zero coefficients.
     */
    [[deprecated("Use make_zero_matrix_like() instead.")]]
    static constexpr auto Zero(const Index r, const Index c)
    {
      return make_zero_matrix_like<Derived>(Dimensions{static_cast<std::size_t>(r)}, Dimensions{static_cast<std::size_t>(c)});
    }


    /**
     * \note Overrides Eigen::DenseBase<Derived>::Identity.
     * \return An identity matrix with the same or identified number of rows and columns.
     */
    [[deprecated("Use make_identity_matrix_like() instead.")]]
    static constexpr auto Identity()
    {
      if constexpr(square_matrix<Derived>)
        return make_identity_matrix_like<Derived>();
      else
        return Base::Identity();
    }


    /**
     * \note Overrides Eigen::DenseBase<Derived>::Identity.
     * \return An identity matrix with the same or identified number of rows and columns.
     */
    [[deprecated("Use make_identity_matrix_like() instead.")]]
    static constexpr decltype(auto) Identity(const Index r, const Index c)
    {
      if constexpr (not dynamic_dimension<Derived, 0>) if (r != index_dimension_of_v<Derived, 0>)
        throw std::invalid_argument {"In T::Identity(r, c), r (==" + std::to_string(r) +
        ") does not equal the fixed rows of T (==" + std::to_string(index_dimension_of_v<Derived, 0>) + ")"};

      if constexpr (not dynamic_dimension<Derived, 1>) if (c != index_dimension_of_v<Derived, 0>)
        throw std::invalid_argument {"In T::Identity(r, c), c (==" + std::to_string(c) +
        ") does not equal the fixed columns of T (==" + std::to_string(index_dimension_of_v<Derived, 1>) + ")"};

      return make_identity_matrix_like<Derived>(static_cast<std::size_t>(r), static_cast<std::size_t>(c));
    }


  private:

    template<typename Arg>
    constexpr auto& get_ultimate_nested_matrix_impl(Arg& arg)
    {
      auto& b = nested_matrix(arg);
      using B = decltype(b);
      static_assert(not std::is_const_v<std::remove_reference_t<B>>);
      if constexpr(eigen_self_adjoint_expr<B> or eigen_triangular_expr<B> or
        eigen_diagonal_expr<B> or euclidean_expr<B>)
      {
        return get_ultimate_nested_matrix(b);
      }
      else
      {
        return b;
      }
    }


    template<typename Arg>
    constexpr auto& get_ultimate_nested_matrix(Arg& arg)
    {
      if constexpr(eigen_self_adjoint_expr<Arg>)
      {
        if constexpr (hermitian_adapter_type_of_v<Arg> == TriangleType::diagonal) return arg;
        else return get_ultimate_nested_matrix_impl(arg);
      }
      else if constexpr(eigen_triangular_expr<Arg>)
      {
        if constexpr(triangular_matrix<Arg, TriangleType::diagonal>) return arg;
        else return get_ultimate_nested_matrix_impl(arg);
      }
      else
      {
        return get_ultimate_nested_matrix_impl(arg);
      }
    }


  public:

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto operator<<(const S& s)
    {
      if constexpr(covariance<Derived>)
      {
        auto& xpr = static_cast<Derived&>(*this);
        return Eigen::CovarianceCommaInitializer {xpr, static_cast<const Scalar&>(s)};
      }
      else
      {
        auto& xpr = get_ultimate_nested_matrix(static_cast<Derived&>(*this));
        using Xpr = std::decay_t<decltype(xpr)>;
        if constexpr(mean<Derived>)
        {
          return Eigen::MeanCommaInitializer<Derived, Xpr> {xpr, static_cast<const Scalar&>(s)};
        }
        else if constexpr((eigen_self_adjoint_expr<Xpr> or eigen_triangular_expr<Xpr>)
          and diagonal_matrix<Xpr>)
        {
          return Eigen::DiagonalCommaInitializer {xpr, static_cast<const Scalar&>(s)};
        }
        else
        {
          return Eigen::CommaInitializer {xpr, static_cast<const Scalar&>(s)};
        }
      }
    }


#ifdef __cpp_concepts
    template<indexible Other>
#else
    template<typename Other, std::enable_if_t<indexible<Other>, int> = 0>
#endif
    auto operator<<(const Other& other)
    {
      if constexpr(covariance<Derived>)
      {
        auto& xpr = static_cast<Derived&>(*this);
        return Eigen::CovarianceCommaInitializer {xpr, other};
      }
      else
      {
        auto& xpr = get_ultimate_nested_matrix(static_cast<Derived&>(*this));
        using Xpr = std::decay_t<decltype(xpr)>;
        if constexpr (mean<Derived>)
        {
          return Eigen::MeanCommaInitializer<Derived, Xpr> {xpr, other};
        }
        else if constexpr ((eigen_self_adjoint_expr<Xpr> or eigen_triangular_expr<Xpr>)
          and diagonal_matrix<Xpr>)
        {
          return Eigen::DiagonalCommaInitializer {xpr, other};
        }
        else
        {
          return Eigen::CommaInitializer {xpr, other};
        }
      }
    }

  };

} // namespace OpenKalman::Eigen3


#endif //OPENKALMAN_EIGENADAPTERBASE_HPP
