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


#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct PacketScalarImpl { using type = scalar_type_of<NestedMatrix>; };


#ifdef __cpp_concepts
    template<typename T> requires requires { typename T::PacketScalar; }
    struct PacketScalarImpl<T>
#else
    template<typename T>
    struct PacketScalarImpl<T, std::void_t<typename T::PacketScalar>>
#endif
      { using type = typename T::PacketScalar; };


  public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(bool {has_dynamic_dimensions<NestedMatrix>})


    EigenAdapterBase() = default;

    EigenAdapterBase(const EigenAdapterBase&) = default;

    EigenAdapterBase(EigenAdapterBase&&) = default;

    ~EigenAdapterBase() = default;

    constexpr EigenAdapterBase& operator=(const EigenAdapterBase& other) { return *this; }

    constexpr EigenAdapterBase& operator=(EigenAdapterBase&& other) { return *this; }


    /// \internal \note Eigen3 requires this.
    using Scalar = scalar_type_of_t<NestedMatrix>;

    using PacketScalar = typename PacketScalarImpl<Derived>::type;


    /* \internal
     * \brief The underlying numeric type for composed scalar types.
     * \details In cases where Scalar is e.g. std::complex<T>, T were corresponding to RealScalar.
     */
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;


    /* \internal
     * \brief The return type for coefficient access.
     * \details Depending on whether the object allows direct coefficient access (e.g. for a MatrixXd), this type is
     * either 'const Scalar&' or simply 'Scalar' for objects that do not allow direct coefficient access.
     */
    using typename Base::CoeffReturnType;

    /**
     * \internal
     * \brief The type of *this that is used for nesting within other Eigen classes.
     * \note Eigen3 requires this as the type used when Derived is nested.
     */
    using Nested = Derived;

    using StorageKind [[maybe_unused]] = typename Eigen::internal::traits<Derived>::StorageKind;

    using StorageIndex [[maybe_unused]] = typename Eigen::internal::traits<Derived>::StorageIndex;

    using Index = typename Base::Index;


    enum CompileTimeTraits
    {
      RowsAtCompileTime [[maybe_unused]] = Eigen::internal::traits<Derived>::RowsAtCompileTime,
      ColsAtCompileTime [[maybe_unused]] = Eigen::internal::traits<Derived>::ColsAtCompileTime,
      Flags [[maybe_unused]] = Eigen::internal::traits<Derived>::Flags,
      SizeAtCompileTime [[maybe_unused]] = Base::SizeAtCompileTime,
      MaxSizeAtCompileTime [[maybe_unused]] = Base::MaxSizeAtCompileTime,
      IsVectorAtCompileTime [[maybe_unused]] = Base::IsVectorAtCompileTime,
    };


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

    /**
     * \brief Synonym for zero().
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
     * \brief Synonym for zero().
     * \note Overrides Eigen::DenseBase<Derived>::Zero.
     * \return A matrix, of the same size and shape, containing only zero coefficients.
     */
    [[deprecated("Use make_zero_matrix_like() instead.")]]
    static constexpr auto Zero(const Index r, const Index c)
    {
      return make_zero_matrix_like<Derived>(Dimensions{static_cast<std::size_t>(r)}, Dimensions{static_cast<std::size_t>(c)});
    }


    /**
     * \brief Synonym for identity().
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
     * \brief Synonym for identity().
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
